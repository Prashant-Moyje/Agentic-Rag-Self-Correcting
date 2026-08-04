"""
Streamlit demo front-end for the agentic RAG system.

Run with:
    streamlit run src/streamlit_app.py

Shows the question -> answer flow AND streams the internal trace
(retrieval, grading, query rewrites, retries) so you can actually
demo the "agentic" behavior live, not just show a final answer box.
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import rag_graph  # noqa: E402
from src.ingestion import ingest  # noqa: E402

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")

st.title("🔎 Agentic RAG — Self-Correcting Retrieval")
st.caption(
    "Retrieves from Pinecone + Milvus in parallel, grades its own results, "
    "and rewrites the query and retries if the context isn't good enough."
)

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "1. **Retrieve** — search both vector DBs, merge results\n"
        "2. **Grade** — an LLM judges each chunk for relevance\n"
        "3. **Decide** — enough good evidence? generate. Otherwise...\n"
        "4. **Rewrite** — reformulate the query and go back to step 1\n"
        "5. **Generate** — answer, grounded only in relevant chunks\n\n"
        "Capped retries fall back to an honest \"not found\" instead of "
        "hallucinating."
    )
    st.divider()
    st.caption(
        "Runs locally via Ollama -- no cloud LLM cost. Upload a PDF below "
        "or use one already ingested via the CLI."
    )

st.subheader("Add a document")

uploaded_file = st.file_uploader("Upload a PDF to add to the knowledge base", type=["pdf"])
ingest_clicked = st.button("Ingest this PDF", disabled=uploaded_file is None)

if ingest_clicked and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner(f"Ingesting {uploaded_file.name} into Pinecone + Milvus..."):
        try:
            ingest(tmp_path)
            st.success(f"{uploaded_file.name} ingested successfully. You can now ask questions about it below.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
        finally:
            os.remove(tmp_path)

st.divider()

st.subheader("Ask a question")

question = st.text_input(
    "Ask a question about your ingested documents",
    placeholder="e.g. What are the key risks mentioned in the report?",
)
run_clicked = st.button("Run Agent", type="primary", disabled=not question)

if run_clicked and question:
    initial_state = {
        "question": question,
        "search_query": question,
        "documents": [],
        "retries": 0,
        "answer": "",
    }

    trace_container = st.container()
    trace_container.subheader("Live agent trace")

    final_state = None
    step_num = 0

    with st.spinner("Running the agent..."):
        for state in rag_graph.stream(initial_state, stream_mode="values"):
            step_num += 1
            final_state = state

            with trace_container.expander(
                f"Step {step_num} — query: {state.get('search_query', question)!r} "
                f"| retries so far: {state.get('retries', 0)} "
                f"| relevant docs: {len(state.get('documents', []))}",
                expanded=(step_num == 1),
            ):
                if state.get("documents"):
                    for i, doc in enumerate(state["documents"], 1):
                        st.markdown(f"**Doc {i}:**")
                        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                else:
                    st.write("No documents in state yet.")

    st.divider()
    st.subheader("Final Answer")

    if final_state and final_state.get("answer"):
        st.markdown(final_state["answer"])

        if final_state.get("documents"):
            with st.expander(f"📚 Sources used ({len(final_state['documents'])} chunks)"):
                for i, doc in enumerate(final_state["documents"], 1):
                    st.markdown(f"**Source {i}**")
                    st.text(doc.page_content)
                    if doc.metadata:
                        st.caption(str(doc.metadata))
    else:
        st.warning("No answer was produced.")

    retries_used = final_state.get("retries", 0) if final_state else 0
    if retries_used > 0:
        st.info(
            f"🔁 The agent rewrote its search query {retries_used} time(s) "
            "because the initial retrieval wasn't relevant enough — "
            "this is the self-correction loop in action."
        )