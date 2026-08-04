"""
CLI entrypoint.

Usage:
    python -m src.app "your question here"
"""
import sys

from .graph import rag_graph


def ask(question: str) -> str:
    initial_state = {
        "question": question,
        "search_query": question,
        "documents": [],
        "retries": 0,
        "answer": "",
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["answer"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.app "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(ask(question))