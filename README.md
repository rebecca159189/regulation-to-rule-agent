# Regulation-to-Rule Assistant (LangGraph agent + hybrid RAG)

[中文说明](README.zh-CN.md)

Turns a regulatory notice into a **disabled, human-reviewed rule-engine configuration**:
read the notice → retrieve the relevant clauses and platform fields → an LLM drafts a structured rule through **tool calls** → **deterministic code** validates every element → a person reviews → export the rule with `is_enabled = 0`.

The core idea: **the model reads, the code decides.** The model can only emit a constrained, schema-checked draft. It never writes expressions, SQL or code; the final rule expression is compiled by the program, and every rule ships disabled until a human approves it.

> Personal prototype built from my experience on a bank's transaction anti-fraud platform, where turning a regulatory notice into a rule meant manually checking fields, amount units, watch-list versions and effective dates. All platform fields, lists and IDs in this repo are **generic demo data**, not any production configuration.

## What it does

- **Document ingestion**: PDF / DOCX / TXT / MD, split into clauses and chunks.
- **Hybrid retrieval**: jieba-tokenised BM25 + `bge-small-zh-v1.5` embeddings in Chroma, fused with Reciprocal Rank Fusion; three isolated scopes (this notice / platform schema / existing rules) via metadata filters.
- **Agent loop (LangGraph `StateGraph`)**: the model has 6 tools — three searches, `ask_human`, `submit_draft`, `handoff_to_human` — and a call budget.
- **Guardrails in code** (`regrule/validation.py`): verbatim quote check against the source text, comparison-operator wording (e.g. "不低于" must be `>=`), exact yuan/fen conversion in integer arithmetic (no floats), list-version check, time-zone-aware effective dates, clause coverage, and boundary tests on every threshold. Errors go back to the model as structured feedback so it can fix its own draft.
- **Human in the loop**: `interrupt()` pauses the graph when the model needs to ask a question or when a draft is ready for review; state is checkpointed in SQLite (`SqliteSaver`), so a session survives a page reload. Approvals are bound to a content fingerprint — editing the draft invalidates the approval.
- **Refuses what it can't do**: a rolling-window rule ("cumulative amount in 24 hours") is marked *blocked* rather than approximated; a notice with a missing unit or list version triggers a question instead of a guess.

```
                 ┌──────── validation failed / more retrieval ─────────┐
                 ▼                                                      │
 START → prepare → agent ──(tool call)──► tools ────────────────────────┤
                    │                       ├─ ask_human ──► clarify ⏸ wait for answer ─► agent
                    │ budget exhausted      │
                    ▼                       └─ draft valid / handed off ─► review ⏸ wait for reviewer
                   END                                                     ├─ edit → re-validate
                                                                           ├─ send back → agent
                                                                           ├─ reject → END
                                                                           └─ approve → export → END
```

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add an OpenAI-compatible key (e.g. DeepSeek) for real-model mode
streamlit run app.py          # http://localhost:8501
```

Without a key the app runs in **demo mode**: a scripted model submits hand-written reference drafts so you can walk through retrieval → validation → review → export. Demo mode is not a model and says so in the UI.

Things to try: load example "01 大于3000元", run the agent, then on the review tab change the operator from `gt` to `gte` and re-validate — you'll see `OPERATOR_SOURCE` and `BOUNDARY` errors.

## Tests and evaluation

```bash
python -m pytest -q                  # 45 tests, offline: fake model + mocked OpenAI endpoint
python scripts/eval.py --mode demo   # pipeline check on 10 labelled examples: 10/10
python scripts/eval.py --mode llm    # same 10 examples against a real model (uses tokens)
```

The 10 hand-written examples cover operator wording (`>`, `>=`, `<`, `<=`, `=`), decimals, amounts already in fen, and three traps: missing unit (should ask), missing list version (should ask), rolling window (should be blocked). `--mode demo` only proves the pipeline works end to end; real-model accuracy has to be measured with `--mode llm`.

## Project layout

```
app.py                     Streamlit UI
regrule/
  schemas.py               Pydantic models (draft schema is sent to the model as JSON Schema)
  documents.py             loading and clause / chunk splitting
  embeddings.py            local bge-small-zh or an OpenAI-compatible embedding API
  retrieval.py             BM25 + Chroma + RRF
  validation.py            deterministic checks and rule compilation
  platform.py              demo platform profile (fields, lists, templates)
  fixtures.py              10 labelled examples
  store.py, service.py     SQLite storage and the service layer
  llm.py                   ChatOpenAI client for any OpenAI-compatible API
  agent/graph.py           LangGraph workflow
  agent/tools.py           the 6 tools
  agent/prompts.py         system and context prompts
  agent/demo_model.py      scripted model for demo mode
scripts/eval.py            batch evaluation
tests/                     unit tests
```

## Limitations

- Demo-mode results are not model results; run `eval.py --mode llm` for real accuracy.
- No reranker and no OCR (scanned PDFs are not supported).
- A human answer does not replace evidence in the source: if the notice omits the unit, answering "yuan" still won't pass validation. This is deliberate.
- The platform profile is demo data and has not been loaded into a real rule engine.
- Single-user, local only; no auth and no tracing (e.g. LangSmith).

## Tech stack

Python · LangGraph · LangChain (core / OpenAI / text-splitters / Chroma) · Chroma · fastembed (`BAAI/bge-small-zh-v1.5`) · rank-bm25 · jieba · Pydantic · Streamlit · SQLite · pytest
