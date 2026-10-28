# AI-Assisted Data Labeling Tool

An LLM suggests a label for each item, a human accepts or fixes it, and the tool
measures how far the AI can be trusted. The default task labels the genre of
200 music-clip captions from Google's **MusicCaps** dataset.

```
CSV ──import──▶ SQLite ──LLM suggests──▶ review UI (accept / fix) ──▶ metrics + CSV export
```

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

streamlit run app.py
```

In the sidebar, click **Load MusicCaps sample**, then **Label 200 items** with the
LLM. Review in the **Review** tab, read the results in **Metrics**, and
download the CSV from **Export**.

No API key? Choose **Keyword baseline** in the sidebar. It runs offline and is
also the baseline the LLM should beat.

The same steps run from the command line:

```bash
python -m labeler import data/musiccaps_sample.csv
python -m labeler suggest                    # LLM; --labeler keyword for the baseline
python -m labeler stats                      # after reviewing in the UI
python -m labeler export -o labels.csv       # reviewed items; --all for everything
```

Every command is safe to re-run. `suggest` only labels items that have no
suggestion yet, so an interrupted run picks up where it stopped.

## What it measures

The human label is the ground truth. Only reviewed items are counted.

| Metric | Question it answers |
|---|---|
| Accuracy | How often did the human keep the AI's label? |
| Cohen's kappa | How much of that agreement is beyond chance? (0 = chance, 1 = perfect) |
| Precision / recall per label | Which genres does the AI over-predict or miss? |
| Confusion matrix | Which genres get mixed up with which? |
| Accuracy by confidence band | Is the AI's confidence honest? If it is, high-confidence items need less checking. |
| Disagreements table | The exact items where the AI was wrong, to read and learn from. |

The review queue shows the **least confident items first**, so human time goes
where the AI is most likely to be wrong.

## Design choices

- **The AI can't invent labels.** The LLM answers through structured output whose
  schema lists the allowed labels as an `enum`. The response is validated again
  in code, and anything off-schema, refused or cut off is skipped and left pending.
- **No lost work.** One SQLite row per item, committed on every write.
- **The dataset can't steer the labeler.** Each caption is sent inside
  `<caption>` tags, and the prompt says to treat it only as data.
- **Failures stay local.** Rate limits and server errors are retried by the SDK.
  If an item still fails, it is reported and the run continues. A bad API key
  stops the run at once instead of failing 200 times.
- **One file defines the task.** To label something else (sentiment, mood,
  topic), edit the labels and guidelines in `labeler/task.py` and import any CSV
  with `id,text` columns.

Model: `claude-opus-5-5` at low effort. Set `LABELER_MODEL` to use another model.

## Layout

```
labeler/task.py       labels + guidelines
labeler/store.py      SQLite storage
labeler/labelers.py   LLMLabeler, KeywordLabeler
labeler/pipeline.py   parallel pre-labeling with per-item failure handling
labeler/metrics.py    accuracy, kappa, per-label, confusion, confidence bands
labeler/export.py     CSV export
labeler/cli.py        command line
app.py                Streamlit review UI
tests/                37 tests: metrics, storage, LLM response parsing, pipeline, CLI, UI
SPEC.md               spec and implementation plan
```

## Tests

```bash
python -m pytest -q
```

The tests don't need an API key. LLM responses are faked, including refusals,
off-schema labels and rate-limit errors.

## Data

`data/musiccaps_sample.csv` is 200 captions sampled from MusicCaps (Google,
CC BY-SA 4.0). See `data/ATTRIBUTION.md`.
