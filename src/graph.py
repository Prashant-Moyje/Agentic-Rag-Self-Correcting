"""Builds the LangGraph state machine: retrieve -> grade -> (generate | rewrite -> retrieve)."""
from langgraph.graph import StateGraph, END

from .graph_state import RagState
from .graph_nodes import retrieve, grade_documents, decide_next_step, rewrite_query, generate, give_up


def build_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "give_up": "give_up",
        },
    )

    # the self-correcting loop: rewritten query goes back through retrieval
    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_edge("generate", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()


rag_graph = build_graph()