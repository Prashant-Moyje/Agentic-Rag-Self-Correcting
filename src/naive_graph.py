"""
A 'naive' RAG baseline for comparison: retrieve once, generate directly.
No grading, no query rewriting, no retry loop. Used only to measure how
much the self-correction loop in graph.py actually helps.
"""
from langgraph.graph import StateGraph, END

from .graph_state import RagState
from .graph_nodes import retrieve, generate


def build_naive_graph():
    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


naive_rag_graph = build_naive_graph()