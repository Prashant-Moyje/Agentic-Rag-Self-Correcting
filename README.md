\# Agentic RAG: Self-Correcting Retrieval Across Pinecone + Milvus



An agentic Retrieval-Augmented Generation system built with \*\*LangGraph\*\*. Unlike

a standard RAG pipeline (retrieve once → generate), this agent:



1\. \*\*Retrieves in parallel\*\* from two vector databases (Pinecone + Milvus)

2\. \*\*Grades its own retrieved documents\*\* for relevance using an LLM judge

3\. \*\*Rewrites the search query\*\* and retries if the context isn't good enough

4\. \*\*Generates a grounded answer\*\* only once it has sufficient relevant context,

&#x20;  or honestly says it couldn't find an answer after max retries



This pattern is called \*\*Corrective RAG (CRAG)\*\* — worth knowing that name, it's

the standard industry term for this kind of self-correcting retrieval loop.



\## Architecture

&#x20;       ┌─────────────┐



question ────────► │ retrieve │ (Pinecone + Milvus, parallel, merged)

└──────┬──────┘

▼

┌─────────────┐

│ grade │ LLM judges each doc: relevant? yes/no

└──────┬──────┘

▼

enough relevant docs?

│ │

yes no (and retries left)

│ │

▼ ▼

┌────────────┐ ┌──────────────┐

│ generate │ │ rewrite\_query│──► loop back to retrieve

└────────────┘ └──────────────┘

│

▼

answer (with source citations, or an honest

"insufficient evidence" response after max retries)

\## Project layout



agentic-rag/

├── src/

│ ├── config.py # env vars / settings

│ ├── vector\_stores.py # Pinecone + Milvus retriever wrappers

│ ├── ingestion.py # load PDFs -> chunk -> embed -> upsert to both DBs

│ ├── graph\_state.py # LangGraph state schema

│ ├── graph\_nodes.py # retrieve / grade / rewrite / generate node functions

│ ├── graph.py # wires nodes into the LangGraph StateGraph

│ ├── naive\_graph.py # baseline (no grading/rewrite) for comparison

│ ├── evaluate.py # runs corrective vs naive on 10 test questions

│ ├── app.py # CLI entrypoint

│ └── streamlit\_app.py # live demo UI showing the agent's trace

├── EVALUATION.md # methodology, results, and honest caveats

├── requirements.txt

├── .env.example

└── README.md





\## Setup



1\. \*\*Install dependencies\*\*

```bash

&#x20;  pip install -r requirements.txt

```



2\. \*\*Get accounts\*\*

&#x20;  - Pinecone: https://www.pinecone.io (free tier)

&#x20;  - Milvus: https://cloud.zilliz.com (Zilliz Cloud free tier, managed Milvus)

&#x20;  - Local LLM via \[Ollama](https://ollama.com) (free), or swap in OpenAI's API



3\. \*\*Copy `.env.example` to `.env` and fill in your keys\*\*

```bash

&#x20;  cp .env.example .env

```



4\. \*\*Ingest documents into both vector stores\*\*

```bash

&#x20;  python -m src.ingestion --pdf path/to/your.pdf

```



5\. \*\*Ask questions (CLI)\*\*

```bash

&#x20;  python -m src.app "What were the key risks mentioned in the report?"

```



6\. \*\*Or run the live demo UI\*\*

```bash

&#x20;  streamlit run src/streamlit\_app.py

```

&#x20;  Shows the whole agent trace step by step — each retrieval, the documents

&#x20;  that passed/failed grading, and any query rewrites — before showing the

&#x20;  final grounded answer with its sources.



7\. \*\*Run the evaluation\*\* (self-correcting vs. naive baseline)

```bash

&#x20;  python -m src.evaluate

```

&#x20;  See `EVALUATION.md` for full methodology and results.



\## Why this is more than the "basic agentic RAG" tutorial version



\- \*\*Grading is a real LLM call\*\*, not a heuristic — you can show the actual

&#x20; grading prompt.

\- \*\*The retry loop is a genuine cycle in the graph\*\* (LangGraph conditional

&#x20; edges), capped at `MAX\_RETRIES`, with a graceful "insufficient evidence"

&#x20; fallback instead of hallucinating.

\- \*\*Two independently-configured vector backends\*\* are queried and merged in

&#x20; parallel — a realistic enterprise scenario.

\- \*\*Backed by a real evaluation\*\* against a naive baseline, not just a demo

&#x20; that happens to work (see `EVALUATION.md`).







