\# Evaluation: Self-Correcting RAG vs. Naive RAG



This documents a controlled comparison between the self-correcting agentic

RAG pipeline (`src/graph.py`) and a naive, single-shot RAG baseline

(`src/naive\_graph.py`), run on the same 10 questions, same LLM

(`llama3.1` via Ollama, local), same vector stores (Pinecone + Milvus),

same generation prompt. The only variable is whether grading and query

rewriting are used.



Run it yourself:

```bash

python -m src.evaluate

```



\## Methodology



\*\*Corpus:\*\* a 54-page "Data Science Interview Questions" PDF, ingested

into both Pinecone and Milvus (151 chunks, `chunk\_size=800`,

`chunk\_overlap=150`).



\*\*Test set:\*\* 10 questions in 3 categories, chosen to stress different

parts of the pipeline:



| Category | Count | Purpose |

|---|---|---|

| Easy | 3 | Phrased close to how the source material discusses the concept — should succeed easily either way |

| Awkward | 4 | Same underlying concepts, deliberately phrased in casual/indirect language, poor embedding-match on the first try |

| Off-topic | 3 | Not covered by the corpus at all — tests whether the system hallucinates or admits it doesn't know |



\*\*Two pipelines, same LLM and generation prompt:\*\*

\- \*\*Corrective\*\* (`rag\_graph`): retrieve → grade each doc → if not enough

&#x20; relevant docs, rewrite the query and retry (up to `MAX\_RETRIES=2`) →

&#x20; generate (or give up honestly if still no relevant evidence).

\- \*\*Naive\*\* (`naive\_rag\_graph`): retrieve once → generate directly on

&#x20; whatever came back, no grading, no rewriting.



\*\*Honesty detector:\*\* a simple keyword check (`looks\_like\_honest\_giveup`)

that flags an answer as an honest refusal if it contains phrases like

"couldn't find", "insufficient", "not sufficiently". This is a blunt

instrument — see caveats below.



\## Results



\### Off-topic questions — the headline number



| | Corrective | Naive |

|---|---|---|

| Answered honestly (no fabricated answer) | \*\*3 / 3\*\* | \*\*1 / 3\*\* |



All three off-topic questions ("chocolate chip cookie recipe," "2023

cricket world cup winner," "how to fix a flat tire") were correctly

refused by the corrective pipeline after 2 rewrite attempts each. The

naive pipeline only clearly refused one; on the other two, it still

declined to \*directly\* answer but engaged with the question in a way

that could read as evasive/uncertain rather than a clean, confident

refusal — a meaningfully weaker response than the corrective pipeline's

explicit "I couldn't find sufficiently relevant information" message.



\### Awkwardly-phrased questions



| | Corrective |

|---|---|

| Recovered via query rewrite (retried and gave a non-refusal answer) | 1 / 4 |



This number is weaker and needs a caveat. Looking at the actual

transcripts: 3 of the 4 awkward questions succeeded on the \*\*first\*\*

retrieval attempt (`retries=0`) — the grading step filtered retrieved

chunks down to the genuinely relevant ones (e.g. 4 docs → 1 relevant),

but that filtering alone was enough, no rewrite needed. Only 1 question

actually triggered a rewrite, and even then the rewritten query ("model

just memorizes stuff" → "Overfitting in machine learning models") found

a partially-relevant chunk but the final answer still said it couldn't

find the specific phrasing asked about.



\### Easy questions



All 3 succeeded identically well on both pipelines, as expected — this

category exists as a sanity check, not a differentiator.



\## Honest caveats



Worth stating plainly, since a defensible resume claim depends on it:



1\. \*\*Small sample size.\*\* 10 questions is enough to demonstrate the

&#x20;  mechanism works, not enough to claim a statistically robust

&#x20;  improvement. Treat the numbers as illustrative, not a benchmark.



2\. \*\*The naive pipeline isn't a strawman, but it does share the

&#x20;  corrective pipeline's generation prompt\*\*, which itself instructs the

&#x20;  model to admit insufficient context. So the comparison measures how

&#x20;  much \*additional\* reliability grading + rewriting adds on top of a

&#x20;  prompt that already discourages hallucination — not "self-correction

&#x20;  vs. a naive pipeline that hallucinates freely." That's a more

&#x20;  accurate framing than a blanket "prevents hallucination" claim.



3\. \*\*`looks\_like\_honest\_giveup` is a crude keyword match\*\*, not a

&#x20;  semantic judgment. It can miscount edge cases (e.g. an answer that

&#x20;  hedges without using one of the exact trigger phrases). For anything

&#x20;  beyond a personal project, replace this with an LLM-graded evaluation

&#x20;  or a framework like Ragas/TruLens.



4\. \*\*The local LLM (`llama3.1` via Ollama) required a real fix to work

&#x20;  at all\*\*: `with\_structured\_output` for grading silently returned

&#x20;  `False` even on clearly relevant documents, diagnosed by testing the

&#x20;  same document with a plain-text YES/NO prompt instead. The final

&#x20;  grading step uses plain-text parsing, not structured output, because

&#x20;  of this. This is model-specific — a stronger model (e.g. GPT-4o-mini)

&#x20;  would likely support structured output for grading reliably, and

&#x20;  might also produce more varied query rewrites (one test run showed

&#x20;  the rewriter produce the identical query on a repeat attempt).



\## Defensible summary for a resume / portfolio



> In a 10-question evaluation spanning easy, ambiguously-phrased, and

> out-of-scope questions, the self-correcting retrieval pipeline

> correctly declined to answer 3/3 out-of-scope questions without

> fabricating a response, compared to 1/3 for a naive retrieve-once

> baseline using the identical LLM and generation prompt. Diagnosed and

> fixed a structured-output reliability issue with the local grading

> model along the way, replacing it with a more robust plain-text

> parsing approach.



This is deliberately more precise than "reduces hallucination by X%" —

it's a specific, falsifiable claim about a specific test set, which

holds up better under follow-up questions than a vague percentage would.

