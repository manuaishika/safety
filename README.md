# Indic Refusal-Consistency Eval (Inspect)

**Question:** do an LLM's safety refusals hold when the *same* request is written in Hindi or Hinglish instead of English, without also making it refuse harmless requests?

- 21 request pairs x 3 languages (English / Hindi / Hinglish) = 63 samples in `data/seed.jsonl`
- 12 harmful (should refuse) + 9 benign controls (should comply, catches over-refusal)
- Scored by an LLM grader (REFUSE / COMPLY / UNCLEAR); `analyze.py` prints per-language rates and the gap vs English

## Run
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY
python build_dataset.py
inspect eval indic_refusal_eval.py --model anthropic/claude-sonnet-5-5
inspect eval indic_refusal_eval.py --model openai/gpt-4o -T grader_model=anthropic/claude-sonnet-5-5
python analyze.py
inspect view          # browse transcripts
```
Use ONE fixed grader model across all runs so results are comparable.

## 4-week plan
- **Week 1:** run the seed set on 2-3 models; read every transcript by hand; fix bad prompts or translations.
- **Week 2:** grow to 100+ pairs (add Tamil/Bengali/Marathi if you have native checkers); add India-specific categories (caste, communal, UPI/loan-app scams, election misinformation). Hand-label 50 grader outputs and report agreement (Cohen's kappa).
- **Week 3:** add attack styles (roleplay framing, code-switching mid-sentence, transliteration noise); report the gap with confidence intervals (paired design: use discordant pairs / McNemar).
- **Week 4:** write up on LessWrong or the Alignment Forum; publish the repo with a clear README; send to SPAR / Pivotal as your artifact.

## Known limits (state these in your write-up)
- The seed set is tiny and for pipeline validation only; it is not enough for claims.
- LLM grader quality in Hindi/Hinglish needs human validation.
- Translations were written by an AI; have native speakers review.
- Keep harmful prompts low-severity; don't publish model outputs that contain real harmful content.
