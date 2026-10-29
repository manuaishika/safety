# IndicSafe

Measures whether an LLM's refusals of harmful requests, and its willingness to answer benign
ones, hold across English, Hindi, Hinglish and (planned) Tamil, Bengali, Marathi and Telugu.
Runs entirely on free, local, open-weights models. No API keys.

## Use
    uv sync
    uv run indicsafe data validate
    uv run indicsafe data build --languages en,hi,hi-Latn
    make check

## Data
`data/items/<category>/<item_id>.yaml`: one request per file with its translations.
Every harmful item has a paired benign control. Hinglish is `hi-Latn`. All translations are
currently AI-written and unreviewed.

Status: M1 (dataset tooling). Design notes and decisions: `docs/DECISIONS.md`.
