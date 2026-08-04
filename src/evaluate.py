"""
Compares the self-correcting agentic RAG graph against a naive
(retrieve-once, no grading, no rewrite) baseline on the same set of
test questions. Produces the metric to back up the resume claim.

Usage:
    python -m src.evaluate
"""
from .graph import rag_graph
from .naive_graph import naive_rag_graph

TEST_QUESTIONS = [
    ("easy", "What is overfitting?"),
    ("easy", "What is cross-validation?"),
    ("easy", "What is a confusion matrix?"),
    ("awkward", "explain the thing where you use test data vs train data and why models mess up if you don't do it right"),
    ("awkward", "what's that technique called where you check how good your model guesses by splitting data multiple ways"),
    ("awkward", "why does adding too many features sometimes make things worse"),
    ("awkward", "what do you call it when a model just memorizes stuff instead of learning patterns"),
    ("off_topic", "What is the recipe for chocolate chip cookies?"),
    ("off_topic", "Who won the cricket world cup in 2023?"),
    ("off_topic", "How do I fix a flat tire on my car?"),
]


def run_one(graph, question: str) -> dict:
    initial_state = {
        "question": question,
        "search_query": question,
        "documents": [],
        "retries": 0,
        "answer": "",
    }
    return graph.invoke(initial_state)


def looks_like_honest_giveup(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ["couldn't find", "insufficient", "cannot find", "not sufficiently"])


def main():
    results = []

    for category, question in TEST_QUESTIONS:
        print(f"\n=== [{category}] {question}")

        corrective_state = run_one(rag_graph, question)
        naive_state = run_one(naive_rag_graph, question)

        results.append({
            "category": category,
            "question": question,
            "corrective_retries": corrective_state.get("retries", 0),
            "corrective_docs": len(corrective_state.get("documents", [])),
            "corrective_answer": corrective_state.get("answer", ""),
            "naive_docs": len(naive_state.get("documents", [])),
            "naive_answer": naive_state.get("answer", ""),
        })

        print(f"  corrective: retries={corrective_state.get('retries', 0)}, "
              f"relevant_docs={len(corrective_state.get('documents', []))}")
        print(f"  naive:      docs_used={len(naive_state.get('documents', []))} (ungraded)")

    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    for category in ("easy", "awkward", "off_topic"):
        cat_results = [r for r in results if r["category"] == category]
        print(f"\n--- {category.upper()} ---")
        for r in cat_results:
            print(f"\nQ: {r['question']}")
            print(f"  [corrective] retries={r['corrective_retries']} | "
                  f"honest_giveup={looks_like_honest_giveup(r['corrective_answer'])}")
            print(f"  [corrective answer] {r['corrective_answer'][:200]}")
            print(f"  [naive]      honest_giveup={looks_like_honest_giveup(r['naive_answer'])}")
            print(f"  [naive answer]      {r['naive_answer'][:200]}")

    off_topic = [r for r in results if r["category"] == "off_topic"]
    awkward = [r for r in results if r["category"] == "awkward"]

    corrective_honest = sum(1 for r in off_topic if looks_like_honest_giveup(r["corrective_answer"]))
    naive_honest = sum(1 for r in off_topic if looks_like_honest_giveup(r["naive_answer"]))
    recovered = sum(1 for r in awkward if r["corrective_retries"] > 0 and not looks_like_honest_giveup(r["corrective_answer"]))

    print("\n" + "=" * 100)
    print("SUMMARY (this is your resume metric)")
    print("=" * 100)
    print(f"Off-topic Qs answered honestly (no hallucination) -- corrective: {corrective_honest}/{len(off_topic)}")
    print(f"Off-topic Qs answered honestly (no hallucination) -- naive:      {naive_honest}/{len(off_topic)}")
    print(f"Awkwardly-phrased Qs recovered via query rewrite (corrective only): {recovered}/{len(awkward)}")
    print("=" * 100)


if __name__ == "__main__":
    main()