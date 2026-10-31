# IndicSafe

Measures whether an LLM's refusals of harmful requests, and its willingness to answer benign
ones, hold across English, Hindi, Hinglish and (planned) Tamil, Bengali, Marathi and Telugu.
Runs entirely on free, local, open-weights models. No API keys.

## Use
    uv sync
    uv run indicsafe data validate
    uv run indicsafe doctor                          # what fits on this machine
    uv run indicsafe run --config configs/smoke_scripted.yaml --dry-run
    uv run indicsafe run --config configs/smoke_scripted.yaml
    make check

Real models: copy `configs/local_small.example.yaml`, fix the model ids, `uv sync --extra local`,
then `indicsafe bench` and `indicsafe run --max-hours 4`. Re-running a command resumes it.

No GPU: open `notebooks/indicsafe_free_gpu.ipynb` on Kaggle or Colab (runs `scripts/bootstrap_free_gpu.sh`).

## Data
`data/items/<category>/<item_id>.yaml`: one request per file with its translations.
Every harmful item has a paired benign control. Hinglish is `hi-Latn`. All translations are
currently AI-written and unreviewed.

Status: M3b (dataset, variants, generation runs, free-GPU bootstrap). Design notes and decisions: `docs/DECISIONS.md`.
