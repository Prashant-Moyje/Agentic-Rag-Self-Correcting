# Agentic RAG: Self-Correcting Retrieval Across Pinecone + Milvus

**A RAG system that catches its own bad retrievals.** When the documents it pulls back don't actually answer your question, it rewrites the query and searches again — and if it still comes up empty, it says so instead of making something up.

Standard RAG retrieves once and generates from whatever came back. If retrieval was bad — awkward phrasing, a missed embedding, a topic the corpus doesn't cover — the model still produces a confident answer built on irrelevant context. That's where hallucinations come from.

This agent adds a feedback loop:

1. **Retrieves in parallel** from two vector databases (Pinecone + Milvus).
2. **Grades each document** for relevance using an LLM as judge — not a similarity-score cutoff.
3. **Rewrites the query and retries** if too few documents pass grading.
4. **Generates a cited, grounded answer** — or returns "insufficient evidence" once retries are exhausted.

This is the **Corrective RAG (CRAG)** pattern, implemented with real LangGraph conditional edges rather than a loop.

> **Result:** across 10 test questions, this pipeline correctly refused **3/3** out-of-scope questions without fabricating an answer. A naive retrieve-once baseline — same LLM, same prompt — managed **1/3**. [Methodology and caveats →](EVALUATION.md)

---

## Architecture

```text
                    ┌─────────────┐
Question ─────────► │  Retrieve   │
                    │ Pinecone +  │
                    │   Milvus    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Grade    │
                    │ LLM judges  │
                    │ relevance   │
                    └──────┬──────┘
                           │
                 Enough relevant docs?
                    │             │
                  Yes            No
                    │      (Retries left?)
                    │             │
                    ▼             ▼
             ┌────────────┐ ┌──────────────┐
             │  Generate  │ │ Rewrite Query│
             └──────┬─────┘ └──────┬───────┘
                    │              │
                    └──────────────┘
                           ▲
                           │
                      Retry Retrieval

                           ▼
        Final answer with source citations
        or "Insufficient evidence" after
               maximum retries.
```

---

## Quickstart

```bash
python3.12 -m venv venv && source venv/bin/activate   # Windows: py -3.12 -m venv venv
pip install -r requirements.txt

cp .env.example .env        # then fill in your Pinecone + Milvus credentials
ollama pull llama3.1        # local LLM, no cloud cost

python -m src.ingestion --pdf path/to/your.pdf
streamlit run src/streamlit_app.py
```

---

## Setup

### 1. Requirements

- **Python 3.12.** Python 3.14 breaks the dependency stack — several packages in the LangChain/Pinecone ecosystem have no wheels for it yet, and pip silently resolves to years-old incompatible versions. See [BUILD_LOG.md](BUILD_LOG.md) for the full diagnosis.
- **[Pinecone](https://www.pinecone.io/)** — free tier is sufficient.
- **[Zilliz Cloud](https://zilliz.com/cloud)** (managed Milvus) — free tier is sufficient. A local Milvus via Docker also works.
- **[Ollama](https://ollama.com/)** for the local LLM.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then populate your API keys and configuration values.

### 4. Ingest your documents

```bash
python -m src.ingestion --pdf path/to/your.pdf
```

This loads the PDF, splits it into chunks, generates embeddings, and stores them in both Pinecone and Milvus.

### 5. Ask questions from the CLI

```bash
python -m src.app "What were the key risks mentioned in the report?"
```

### 6. Launch the Streamlit interface

```bash
streamlit run src/streamlit_app.py
```

The UI visualizes the complete execution trace: retrieval from both vector databases, LLM relevance grading, query rewrites where they happen, and the final grounded answer with source citations.

### 7. Run the evaluation

```bash
python -m src.evaluate
```

See [EVALUATION.md](EVALUATION.md) for the methodology, benchmark setup, and comparative results.

---

## Project Structure

```
├── src/
│   ├── config.py             # Environment variables and configuration
│   ├── vector_stores.py      # Pinecone & Milvus retriever wrappers
│   ├── ingestion.py          # Load PDFs → chunk → embed → upsert
│   ├── graph_state.py        # LangGraph state schema
│   ├── graph_nodes.py        # Retrieve, grade, rewrite, generate nodes
│   ├── graph.py              # LangGraph workflow definition
│   ├── naive_graph.py        # Baseline RAG (no grading or rewriting)
│   ├── evaluate.py           # Compare corrective vs. naive RAG
│   ├── app.py                # CLI application
│   └── streamlit_app.py      # Interactive visualization UI
├── drop_milvus_collection.py # Utility: drop the Milvus collection to re-ingest cleanly
├── EVALUATION.md             # Methodology and benchmark results
├── BUILD_LOG.md              # Debugging journal from building this
├── requirements.txt
├── .env.example
└── README.md
```

---

## What makes this different from a RAG tutorial

- **LLM-based relevance grading**, not similarity-score thresholding.
- **A real corrective loop** — LangGraph conditional edges route back to retrieval after a query rewrite.
- **Graceful failure** — returns "insufficient evidence" rather than hallucinating when no adequate context is found.
- **Dual vector stores** queried in parallel, mirroring a realistic multi-store enterprise setup.
- **Quantified against a baseline**, with the methodology *and its limitations* both written down in [EVALUATION.md](EVALUATION.md).
- **A debugging journal** in [BUILD_LOG.md](BUILD_LOG.md) documenting the real failures hit along the way — including a silent `with_structured_output` bug in the local grading model that returned false negatives on clearly relevant documents.

---

## Tech Stack

**LangGraph** · **LangChain** · **Pinecone** · **Milvus (Zilliz Cloud)** · **Ollama** · **Streamlit** · **Python 3.12**

---

## License

MIT — see [LICENSE](LICENSE).
