"""
Each function here is one node in the LangGraph. This is where the
'agentic' behavior actually lives: grading its own retrieval, and
deciding whether to rewrite the query and try again.
"""
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .graph_state import RagState
from .vector_stores import multi_store_search

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)

def retrieve(state: RagState) -> RagState:
    query = state.get("search_query") or state["question"]
    docs = multi_store_search(query, k=config.RETRIEVAL_K)
    print(f"[retrieve] query={query!r} -> {len(docs)} docs")
    return {**state, "documents": docs}


grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing whether a retrieved document is relevant "
     "to a user question. Be strict: only say YES if it actually contains "
     "information that helps answer the question. "
     "Answer with exactly one word: YES or NO. No punctuation, no explanation."),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])

grader = grade_prompt | llm


def _parse_grade(response_text: str) -> bool:
    """Parse a plain YES/NO response. Defaults to False (safer than
    silently trusting an unclear response) if the model doesn't clearly say YES."""
    cleaned = response_text.strip().upper()
    return cleaned.startswith("YES")


def grade_documents(state: RagState) -> RagState:
    question = state["question"]
    graded_relevant: List[Document] = []

    for doc in state["documents"]:
        response = grader.invoke({"question": question, "document": doc.page_content})
        if _parse_grade(response.content):
            graded_relevant.append(doc)

    print(f"[grade_documents] {len(graded_relevant)}/{len(state['documents'])} docs judged relevant")
    return {**state, "documents": graded_relevant}


def decide_next_step(state: RagState) -> Literal["generate", "rewrite_query", "give_up"]:
    enough_evidence = len(state["documents"]) >= 1
    retries_left = state["retries"] < config.MAX_RETRIES

    if enough_evidence:
        return "generate"
    if retries_left:
        return "rewrite_query"
    return "give_up"


rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries to improve retrieval from a vector database. "
     "The previous query returned no sufficiently relevant results. "
     "Produce a single improved query -- broader, more specific, or rephrased "
     "with likely synonyms -- and return ONLY the rewritten query text."),
    ("human", "Original question: {question}\nPrevious search query: {search_query}"),
])

rewriter = rewrite_prompt | llm


def rewrite_query(state: RagState) -> RagState:
    new_query = rewriter.invoke({
        "question": state["question"],
        "search_query": state.get("search_query") or state["question"],
    }).content.strip()

    print(f"[rewrite_query] {state.get('search_query')!r} -> {new_query!r}")
    return {**state, "search_query": new_query, "retries": state["retries"] + 1}


generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the provided context. "
     "Cite which source snippet(s) you used. If the context is insufficient, "
     "say so plainly rather than guessing."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

generator = generate_prompt | llm


def generate(state: RagState) -> RagState:
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    answer = generator.invoke({"question": state["question"], "context": context}).content
    return {**state, "answer": answer}


def give_up(state: RagState) -> RagState:
    answer = (
        "I couldn't find sufficiently relevant information in the connected "
        "vector databases to answer this confidently, even after rewriting "
        "the search query. You may want to check whether the right documents "
        "have been ingested, or rephrase the question."
    )
    return {**state, "answer": answer}