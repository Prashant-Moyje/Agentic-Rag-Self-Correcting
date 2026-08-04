# Agentic RAG: Self-Correcting Retrieval Across Pinecone + Milvus

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**. Unlike a standard RAG pipeline (retrieve once → generate), this system performs iterative, self-correcting retrieval to improve answer quality.

The agent:

1. **Retrieves in parallel** from two vector databases (Pinecone + Milvus).
2. **Grades retrieved documents** for relevance using an LLM as a judge.
3. **Rewrites the search query** and retries retrieval if the retrieved context is insufficient.
4. **Generates a grounded answer** only after collecting enough relevant context, or honestly reports that it could not find sufficient evidence after the maximum number of retries.

This pattern is known as **Corrective RAG (CRAG)**, an increasingly common architecture for building self-correcting retrieval systems.

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

## Project Structure

```text
agentic-rag/
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
├── EVALUATION.md             # Methodology and benchmark results
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create required accounts

- **Pinecone** (Free Tier)
- **Zilliz Cloud (Managed Milvus)** (Free Tier)
- **Ollama** for running a local LLM (recommended), or configure the OpenAI API instead.

### 3. Configure environment variables

Copy the example configuration file:

```bash
cp .env.example .env
```

Then populate the required API keys and configuration values.

### 4. Ingest your documents

```bash
python -m src.ingestion --pdf path/to/your.pdf
```

This will:

- Load the PDF
- Split it into chunks
- Generate embeddings
- Store them in both Pinecone and Milvus

### 5. Ask questions from the CLI

```bash
python -m src.app "What were the key risks mentioned in the report?"
```

### 6. Launch the Streamlit interface

```bash
streamlit run src/streamlit_app.py
```

The UI visualizes the complete execution trace, including:

- Retrieval from both vector databases
- LLM relevance grading
- Query rewrites (if needed)
- Final grounded answer with source citations

### 7. Run the evaluation

```bash
python -m src.evaluate
```

See **EVALUATION.md** for the evaluation methodology, benchmark setup, and comparative results.

---

## Why This Project Goes Beyond a Basic Agentic RAG Tutorial

Unlike many introductory RAG examples, this project implements several production-oriented capabilities:

- **LLM-based document grading** rather than simple similarity score filtering.
- **A true corrective retrieval loop** implemented with LangGraph conditional edges, allowing the agent to retry retrieval after rewriting the query.
- **Graceful failure handling**, returning an "Insufficient evidence" response instead of hallucinating when no adequate context is found.
- **Parallel retrieval** from two independently configured vector databases (Pinecone and Milvus), simulating a realistic enterprise retrieval architecture.
- **Quantitative evaluation** against a naive single-pass RAG baseline, with documented methodology and results in `EVALUATION.md`.

---

## Tech Stack

- **LangGraph**
- **LangChain**
- **Pinecone**
- **Milvus (Zilliz Cloud)**
- **Ollama / OpenAI**
- **Streamlit**
- **Python 3.11+**

---

## License

MIT License
