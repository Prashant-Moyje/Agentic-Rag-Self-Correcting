# Build Log: Debugging Journal

This documents real problems hit while building and running this project on
Windows, and how each was diagnosed and fixed. Kept deliberately unpolished —
this is the actual troubleshooting record, useful for interviews ("tell me
about a bug you had to debug") more than a curated success story would be.

## 1. Python 3.14 too new for the dependency stack

**Symptom:** `pip install` fell back to `langchain-pinecone==0.0.1`, which
pulled in `numpy<2`, which has no pre-built wheel for Python 3.14 and tried
to compile from source — failing because no C compiler was present.

**Root cause:** Python 3.14 was released very recently; several packages in
the LangChain/Pinecone ecosystem hadn't published wheels for it yet, so pip
silently resolved to years-old, incompatible versions instead.

**Fix:** Installed Python 3.12 specifically for this project (`py -3.12 -m
venv venv`) rather than using the system default. Once the venv used 3.12,
every package resolved to its current version with no compilation needed.

**Lesson:** when hunting a dependency resolution error, check what Python
version is being targeted before assuming the packages themselves are broken.

## 2. `.env` file silently missing values

**Symptom:** `PineconeConfigurationError: You haven't specified an API key`
even though `.env` existed and config.py loaded default values correctly.

**Root cause:** two separate issues stacked on top of each other —
first, Notepad's save dialog can append `.txt`; second, and the actual
cause here, the `.env` file had been overwritten at some point and only
contained two of the ~12 expected keys.

**Fix:** rebuilt `.env` from scratch, then verified with a one-line script
printing `bool(config.SOME_KEY)` for every required variable before moving
on — catches "file exists but is incomplete" in a way that just checking
`.env` exists doesn't.

## 3. PowerShell quoting breaks multi-line inline Python

**Symptom:** an inline `python -c "..."` command with escaped quotes and a
hyphenated word (`k-fold`) was parsed by PowerShell as a command name
(`CommandNotFoundException: k-fold`), not passed to Python at all.

**Root cause:** PowerShell's quoting/escaping rules differ from bash;
complex multi-line strings with nested quotes don't survive the same way.

**Fix:** switched to writing a small `.py` script file and running
`python script.py` instead of inlining logic in the shell — sidesteps
shell-quoting entirely for anything non-trivial.

## 4. OpenAI `insufficient_quota` (429) on a fresh account

**Symptom:** `RateLimitError: insufficient_quota` on the very first LLM
call, despite a valid API key.

**Root cause:** OpenAI's API billing is separate from a ChatGPT
subscription — new API accounts need a payment method added and don't get
automatic free credits.

**Fix:** rather than pay to keep testing, swapped the LLM entirely to a
local model via **Ollama** (`llama3.1`, free, no rate limits). This
required changing exactly one line (`ChatOpenAI` → `ChatOllama`) since
both expose the same LangChain chat model interface.

## 5. `with_structured_output` silently wrong with a local model

**Symptom:** the grading node marked *every* retrieved document as
irrelevant, including one that was obviously and directly relevant to the
question, causing the agent to always rewrite the query and often give up
even when good context existed.

**Diagnosis process:**
1. Bypassed grading entirely and called `multi_store_search` directly —
   confirmed retrieval itself was returning correct, on-topic chunks.
2. Isolated the grading call on one known-relevant chunk directly —
   `with_structured_output(GradeResult)` returned `relevant=False`.
3. Replaced the structured-output call with the same prompt but asking for
   a plain `YES`/`NO` text answer — got the correct `YES`.

**Root cause:** `llama3.1`'s tool-calling / structured-output support
(used internally by LangChain's `with_structured_output`) is significantly
less reliable than a model like GPT-4o-mini's — it appears to fail or
default silently rather than raising an error, which made the bug much
harder to spot than a crash would have been.

**Fix:** rewrote `grade_documents` to prompt for a plain YES/NO string and
parse it with a simple `.strip().upper().startswith("YES")` check instead
of relying on structured output. More brittle in theory, more reliable in
practice for this model.

**Lesson:** a node that runs without error but produces subtly wrong
results is more dangerous than one that crashes — worth testing intermediate
outputs on known cases, not just checking that the pipeline runs end to end.

## 6. Milvus schema mismatch across different PDFs

**Symptom:** ingesting a second, different PDF after the first had already
been ingested threw `DataNotMatchException: Insert missed a field 'author'`.

**Root cause:** Milvus auto-derives its collection schema from whichever
metadata fields happen to be present in the *first* batch of documents
inserted. The first PDF's metadata happened to include an `author` field;
the second PDF's metadata didn't have one, so its rows didn't match the
now-fixed schema.

**Fix:** stopped passing a PDF's raw, inconsistent metadata straight into
Milvus. Instead, built a `_sanitize_for_milvus` step that always sends the
exact same two fields (`source`, `page`) regardless of what metadata the
source PDF happens to carry — then dropped and recreated the collection
so it wasn't stuck with the old, `author`-requiring schema.

**Lesson:** don't assume different files of "the same type" (PDFs) have
the same metadata shape — PDF producers (Word, PDFTron, LaTeX, scanners)
all embed different metadata fields.

## 7. Windows filename gotchas

**Symptom:** `ValueError: File path ... is not a valid file or url` for a
file that was clearly sitting in the project folder.

**Root cause:** the actual filename had a trailing space before `.pdf`
(`SQL Notes For Beginners .pdf`), invisible when just glancing at Explorer
but present in `dir` output.

**Fix:** always run `dir *.pdf` (or `dir` generally) to get the *exact*
filename before referencing it in a command, rather than typing what looks
right from memory.

## Summary of what these bugs demonstrate

Every one of these was found by **isolating one layer of the pipeline at a
time** (retrieval vs. grading, shell vs. Python, dependency resolution vs.
code logic) rather than assuming the most recently-changed code was at
fault. That's the debugging habit worth naming explicitly in an interview
if this project comes up — the fixes themselves are less interesting than
the process of narrowing down *where* in a multi-step pipeline a failure
actually originates.