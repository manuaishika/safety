# Project: Indic Refusal-Consistency Eval

## Goal
Measure whether an LLM's safety refusals hold when the SAME request is made in Hindi (Devanagari) or Hinglish (Hindi in Roman script) instead of English, while also checking it does not over-refuse harmless requests. Built on UK AISI's **Inspect** (`inspect-ai`). Output is meant to become a public repo plus a LessWrong / Alignment Forum write-up, used as an artifact for AI safety programme applications (Pivotal, SPAR, LASR).

## Design
- **Paired design:** each request exists in 3 languages under one `pair_id`, so refusal is compared like-for-like.
- **Two classes:** `expected="refuse"` (harmful, low-severity) and `expected="comply"` (benign control that touches the same topic, to catch over-refusal).
- **Scoring:** an LLM grader labels each response REFUSE / COMPLY / UNCLEAR. A sample is correct if the label matches `expected`.
- **Key metrics:** harmful-refusal rate per language, benign-compliance rate per language, refusal gap vs English, and pairs that are right in English but wrong in Hindi/Hinglish.

## Files
- `build_dataset.py` : source of truth for prompts (`PAIRS` list); writes `data/seed.jsonl`. Edit this, never hand-edit the jsonl.
- `indic_refusal_eval.py` : Inspect `@task indic_refusal`, grader prompt, `refusal_scorer`.
- `analyze.py` : reads `./logs`, prints per-language tables and English gap.
- `README.md` : run instructions and 4-week plan.

## Commands
```
pip install -r requirements.txt
python build_dataset.py
inspect eval indic_refusal_eval.py --model anthropic/claude-sonnet-5-5
inspect eval indic_refusal_eval.py --model <provider/model> -T grader_model=<fixed grader>
python analyze.py
inspect view
```
Needs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment. Never commit keys or `logs/`.

## Current state
- 21 pairs x 3 languages = 63 samples (12 harmful, 9 benign), categories: stalking, forgery, harassment, fraud, communal and election misinformation, phishing, UPI fraud, loan-app harassment, caste hate, impersonation, academic dishonesty.
- Pipeline verified end-to-end only with a stub model (no API keys in the sandbox; mockllm needs a tiktoken download that was blocked). Not yet run against a real model.
- Hindi/Hinglish text was AI-written and is NOT yet verified by a native speaker.

## Rules
- Harmful prompts stay low-severity (no weapons, CBRN, self-harm methods, sexual content). Do not add content that would give real uplift.
- Do not commit or paste model outputs containing real harmful content.
- Use ONE fixed grader model across all compared runs; record it in the results.
- Never silently change the grader prompt; if you do, bump a `GRADER_VERSION` constant and note it.
- Report uncertainty honestly: small n, grader error, translation quality.

## Roadmap (do in order, ask before big changes)
1. Run the eval on 2-3 models; fix bugs; commit.
2. Add `validate_grader.py`: export 50 graded samples to CSV for hand-labelling, then compute Cohen's kappa vs the grader.
3. Extend `analyze.py`: bootstrap CIs, McNemar test on discordant pairs (English vs Hindi, English vs Hinglish), CSV output.
4. Grow dataset to 100+ pairs; add Tamil / Bengali / Marathi columns (flag for native-speaker review).
5. Add attack-style transforms: roleplay framing, mid-sentence code-switching, transliteration noise; run as separate tasks.
6. Add a results table + figures to the README and draft the write-up in `WRITEUP.md`.
