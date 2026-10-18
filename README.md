# IndicSafe

Evaluation infrastructure for measuring how LLM safety behaviour (refusing harmful requests,
*not* refusing benign ones) varies across Indian languages, scripts and registers: English,
Hindi (Devanagari), Hinglish (Roman script), with Tamil, Bengali, Marathi and Telugu planned.

Zero-cost by design: local open-weights models (Ollama / HF), local graders, free Kaggle/Colab
for GPU runs. No paid API and no API key is ever required.

**Status: M1 (scaffold, schema, validators, dataset tooling).** Inspect tasks, variants,
grading, validation tooling and analysis land in M2 to M7 (see `CLAUDE.md`).

## Quick start
```
uv sync
uv run indicsafe data validate
uv run indicsafe data build --languages en,hi,hi-Latn
uv run indicsafe data stats
make check          # ruff + mypy + pytest (no network, no models)
```

## Data model
`data/items/<category>/<item_id>.yaml` is the single source of truth: one language-independent
item with its translations (`en`, `hi`, `hi-Latn`, ...). `data/built/` is generated and
gitignored. Every harmful item must have a benign control on the same topic
(`paired_item_id`, reciprocal). Language tags are BCP-47; Hinglish is `hi-Latn`.

All current translations are AI-written (`provenance: llm:claude`, `review_status: unreviewed`)
and need native-speaker review before headline results.

## Known limits
- 24 items (12 harmful, 12 benign): pipeline validation only, not enough for claims.
- Dev/test split is hash-based and not frozen yet (`indicsafe data freeze` once the set is big
  enough); see `docs/DECISIONS.md`.
