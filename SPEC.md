# AI-Assisted Data Labeling Tool: Spec and Implementation Plan

## 1. Problem

Annotation work has two costs: the time a person spends labeling, and the errors
that slip into the final dataset. An LLM can pre-label items so the person
mostly confirms instead of typing, but only if we can measure how often the AI
is right and route the person's attention to the items most likely to be wrong.

The tool answers three questions:

1. What does the AI think each item is? (suggestion + confidence + one-line reason)
2. What is the correct label? (a human accepts or fixes every suggestion)
3. How much can we trust the AI? (accuracy, agreement, and where it fails)

## 2. Dataset and task

- **Data:** 200 captions sampled (seed 7) from **MusicCaps** (Google, CC BY-SA 4.0),
  free-text descriptions of 10-second music clips written by professional musicians.
- **Task:** single-label **genre** classification of each caption into
  `rock, pop, electronic, hip_hop, classical, jazz_blues, folk_country, world, other`.
- The task (labels and guidelines) lives in one place, `labeler/task.py`, so a
  different dataset or label set is a one-file change. Any CSV with `id,text`
  columns can be imported.

## 3. Scope

In scope: import CSV, AI pre-labeling, human review UI, metrics, CSV export, tests.

Out of scope (on purpose): multi-user auth, multi-label items, active learning,
model fine-tuning. A single reviewer and a single label per item keep every
metric well defined.

## 4. First principles that drive the design

| Principle | Decision |
|---|---|
| The human label is the ground truth | AI accuracy is computed only on reviewed items; unreviewed items are never counted. |
| Never lose work | SQLite, one row per item, each write committed immediately. Every command is idempotent and resumable. |
| The AI must not invent labels | Structured output with a JSON-schema `enum` of the allowed labels; any response outside it is rejected, not stored. |
| Spend human attention where it matters | Review queue is ordered by AI confidence, lowest first. |
| Measure against a baseline | A zero-cost keyword labeler is included, so the LLM's accuracy has something to beat, and the tool runs offline. |
| Accuracy alone can mislead | Report Cohen's kappa (agreement beyond chance), per-label precision/recall, a confusion matrix, and accuracy by confidence band (is the confidence honest?). |
| Keep the core pure | Metrics and export are plain functions over rows; the UI and CLI are thin shells. |

## 5. Data model (SQLite, one table)

```
items(
  id            TEXT PRIMARY KEY,
  text          TEXT NOT NULL,
  ai_label      TEXT,      -- NULL until suggested
  ai_confidence REAL,      -- 0..1
  ai_rationale  TEXT,
  ai_model      TEXT,      -- which labeler produced it
  human_label   TEXT,      -- NULL until reviewed
  reviewed_at   TEXT       -- ISO-8601 UTC
)
```

Item status is derived, never stored: `pending` (no AI label), `suggested`, `reviewed`.
One table is enough because each item has at most one suggestion and one review.

## 6. Components

```
labeler/
  task.py      labels + guidelines (the only task-specific file)
  store.py     SQLite access: import, read queue, save suggestion, save review
  labelers.py  LLMLabeler (Anthropic API, structured output) and KeywordLabeler (baseline)
  pipeline.py  suggest_all(): label every pending item, commit one at a time, collect failures
  metrics.py   accuracy, Cohen's kappa, per-label P/R/F1, confusion matrix, confidence bands
  export.py    labels -> CSV
  cli.py       python -m labeler {import,suggest,stats,export}
app.py         Streamlit UI: Review / Metrics / Export
```

## 7. Behaviour

- **import:** reads `id,text`; inserts new ids, skips existing ones (re-import is safe);
  rejects blank ids/text and duplicate ids within the file.
- **suggest:** labels only `pending` items (or `--limit N`). One API call per item.
  The SDK retries 429/5xx; a failure on one item is logged and skipped, never fatal.
  A refusal or an off-schema answer counts as a failure.
- **review (UI):** shows the item, the AI label, confidence and reason. One click to
  accept; a select box to fix. A "re-review" mode shows already-reviewed items.
- **stats:** `n_reviewed`, `accuracy`, `kappa`, per-label table, confusion matrix,
  accuracy per confidence band (`<0.6`, `0.6-0.8`, `0.8-0.9`, `>=0.9`).
- **export:** `id,text,label,ai_label,ai_confidence,ai_correct,reviewed_at`
  for reviewed items only (the default), or all items with `--all`.

## 8. LLM call

- Model `claude-opus-5-5` (override with `LABELER_MODEL`), effort `low`
  (classification does not need deep reasoning), `max_tokens` 1024.
- `output_config.format` = JSON schema `{label: enum, confidence: number, rationale: string}`.
- Server-side refusal fallback enabled (`fallbacks: "default"`).
- The caption is passed inside `<caption>` tags and the system prompt says to treat
  it as data, so text in the dataset cannot steer the labeler.

## 9. Testing

- Unit: metrics against hand-computed values (incl. kappa edge cases), store
  idempotency and validation, export columns, labeler parsing with a fake client
  (valid, off-schema label, refusal, out-of-range confidence).
- Integration: CLI end to end on a temp DB with the keyword labeler
  (import -> suggest -> review -> stats -> export).
- UI smoke test with Streamlit's `AppTest`.

## 10. Implementation plan

1. Task definition and sample dataset (done first so everything else has real data).
2. `store.py` + tests.
3. `metrics.py` + tests (pure, easiest to get exactly right).
4. `labelers.py` (LLM + keyword) + tests with a fake client.
5. `pipeline.py`, `export.py`, `cli.py` + end-to-end test.
6. `app.py` Streamlit UI + AppTest smoke test.
7. README with quickstart, screenshots of results, and resume bullets.
