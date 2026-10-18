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
M1 scaffold/schema/validators/migration (done) | M2 variants | M3 Inspect tasks, configs, RunMeta,
doctor/bench/planner, checkpointing | M3b first local run + Kaggle/Colab bootstrap | M4 local grading
stack (rubric, ensemble, lexicon, optional IndicTrans2) | M5 human validation (mandatory ground
truth) | M6 analysis (cluster bootstrap over item_id, exact McNemar + Holm, mixed model, power,
figures) | M7 hardening + reproducibility. Each ends with tests + lint + types green and a commit.

## Conventions
Decisions and open questions live in `docs/DECISIONS.md`. Commits are authored as the researcher
(`manuaishika <manuaishika@gmail.com>`).
