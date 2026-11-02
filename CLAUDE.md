# IndicSafe: working notes for Claude

Multilingual safety-behaviour eval infrastructure (see README). Researcher designs the study and
writes the paper; Claude does the engineering. Numbers must be defensible enough for a paper.

## Hard rules
- **Zero cost:** no paid APIs, no required keys, ever. Subject and grader models are local
  open-weights (Ollama / HF), IDs only in `configs/`. Hosted free tiers are optional, off by
  default, rate-limited, and never needed for CI.
- **Safety:** harmful items stay low/medium severity (no weapons, CBRN, self-harm methods, sexual
  content with minors, operational uplift). Raw harmful-compliance output stays local; public
  exports redact response text by default. Test-split text is not published until the researcher
  says so. Never commit keys, `logs/`, `results/`, `data/built/`.
- **Frozen test split:** never modify without `indicsafe data unfreeze --reason ...`.
- **Ask before changing** the schema, rubric, split rule or statistical methods once agreed.
- Never silently change the grader rubric: bump `GRADER_VERSION`; never pool across versions.
- Fail loudly; no silent fallbacks. Report limits honestly (n per cell, grader error, review status).

## Stack
Python 3.11+, uv, src layout (`src/indicsafe/`), pydantic v2, typer, ruff, mypy strict, pytest +
hypothesis. Later milestones add inspect-ai, pandas/pyarrow, scipy/statsmodels, matplotlib.
`make check` = lint + types + tests. CI has no network, no models, no keys: mocked subject and
grader only. mockllm tries to download a tiktoken file; use a custom deterministic ModelAPI stub.

## Milestones
M1 to M7 are built and merged (scaffold, variants, runs/doctor/bench/planner, Colab/Kaggle bootstrap,
local grading, human validation, analysis, hardening). What is left needs people or hardware: a real
local-model run, native-speaker review of translations, human labelling of the validation sample, a
licence for the repo, and the rule for releasing the test split. See `docs/DECISIONS.md`.

## Conventions
Decisions and open questions live in `docs/DECISIONS.md`. Commits are authored as the researcher
(`manuaishika <manuaishika@gmail.com>`).