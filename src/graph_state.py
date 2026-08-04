"""The shared state object that flows through every node in the graph."""
from typing import List, TypedDict
from langchain_core.documents import Document


class RagState(TypedDict):
    question: str            # the original user question (never overwritten)
    search_query: str        # the (possibly rewritten) query used for retrieval
    documents: List[Document]  # documents that passed the relevance grade
    retries: int              # how many times we've rewritten and retried
    answer: str                # final generated answer