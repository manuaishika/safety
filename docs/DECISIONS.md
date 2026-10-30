# Decisions log

## M1
- **Language tags:** BCP-47; Hinglish = `hi-Latn` (script `romanized`). English is script `native` (Latin).
- **Script check:** native Indic text needs >= 50% of letters/marks in its script (Latin acronyms
  such as UPI are legitimate); Latin-language text must be 100% Latin letters.
- **Pairing:** every harmful item needs a reciprocal benign control in the same category. The
  prototype lacked controls for fraud/harassment/impersonation; three were added
  (`spot_fake_reviews`, `report_online_abuse`, `verify_sender_email`), AI-written, unreviewed.
- **Severity tiers (proposed, please review):** harmful = 2 (1 for exam cheating), benign = 1.
- **`india_specific`:** taken from the prototype's second block of items (and their controls).
  `communal_forward`, `bank_phishing_sms` etc. are marked false; revisit.
- **Split:** `sha256(salt:item_id)` threshold (30% test), stable under adding items. Not exactly
  stratified by category (exact stratification would reassign items as categories grow, which a
  frozen test set cannot allow); realised split per category is reported by `data stats`.
  Freezing is deferred until ~100 items: `indicsafe data freeze --reason ...`. After freezing,
  new items always go to dev.
- **`sample_id`:** hash of (item_id, language, variant name, condition) only. Variant versions
  and text are excluded so edits and new variants never change existing ids; `dataset_hash`
  covers content.
- **Unreviewed translations:** carried on every sample as `review_status`. The analysis stage
  (M6) excludes unreviewed by default with an include flag. Note: at present everything is
  unreviewed, so headline tables need that flag until reviewers sign off.
- **Safety denylist** in the validator is a backstop, not a guarantee.
- **Open for M6:** statsmodels has no frequentist logistic GLMM; proposed primary = GEE logistic
  clustered on item_id, sensitivity = `BinomialBayesMixedGLM`.

## M2
- **Variants** are explicit registered instances (`code_switch_25/50/100`, `translit_noise_light/heavy`,
  `register_formal/casual`, `roleplay`, `context_prefix`, `identity`). Parameters live in the name, so
  `sample_id` (names only) stays stable; each carries a semver `version` recorded on the sample.
- **RNG:** per (seed, item_id, language, variant), so build order never matters. Seed is in the manifest.
- **Protected spans:** URLs, emails, handles, numbers/amounts, ALLCAPS acronyms, tokens with digits, and
  non-sentence-initial capitalised Latin words are never altered. The builder re-checks this on every
  sample and fails loudly.
- **Unsupported (variant, language) pairs** raise unless `--skip-unsupported` is given. Noise is
  `hi-Latn` only.
- **Code-switching** uses a small hand-written en<->hi lexicon (~40 entries), so realised ratio is below the
  nominal one and is reported per sample in `variant_info`. Output is not grammatical Hindi.
- **Register** variants wrap the text with a greeting/closing; they do not re-inflect the request.
- **Schema (additive):** `Sample` gains `variant_version`, `variant_info`, `source_text`.
- All wrapper strings and the lexicon are AI-written and need native-speaker review.
- **Planned:** identity-swap variant (needs `{PERSON}` slots in items, so a schema change; will ask first).

## M3
- **Generation and grading are separate steps.** `indic_refusal` stores raw responses (no scorer);
  grading arrives in M4 as `indicsafe grade` over the logs.
- **Run folder / resume:** `run_id` = name + hash(resolved config, dataset hash). Re-running the same
  command reuses the folder and Inspect finishes only unfinished samples. `fail_on_error=True` is
  deliberate: with it off, Inspect records an errored sample as done and never retries it. Logs
  flush after every sample (`log_buffer=1`). Tested for both a failure and a SIGKILL mid-run.
- **One model in memory at a time** (`max_tasks=1`); `max_connections` defaults to 4 (set 1 for
  local models).
- **Hosted providers** are rejected unless `allow_hosted: true`. Providers are recorded in RunMeta.
- **Skipped (variant, language) cells** are allowed by default in configs but always recorded in
  RunMeta and printed by `--dry-run`.
- **Time planner (`--max-hours`):** needs `indicsafe bench` data. Items are kept as harmful+control
  pairs, ordered round-robin across categories. It keeps whole (variant, model) groups, ordered
  variant then model, while at least 8 pair-units fit; if not even one group fits, it trims
  languages of the first group (keeping the first-listed, the baseline). Everything dropped is
  reported. The 8-unit floor is a default, not a statistical power guarantee.
- **Bench token counts** fall back to a length estimate when the provider reports no usage.
- **`doctor` tiers** are rough Q4 memory figures, not measurements. Model tags in
  `configs/local_small.example.yaml` are unverified guesses and must be checked with `ollama list`.
- **CI model double:** `scripted/<policy>` (strict, leaky, overrefuse, compliant) looks up each
  prompt's language and expected behaviour from a registry filled at build time. It avoids
  mockllm's tokenizer download.
- Not yet done: `scripts/bootstrap_free_gpu.sh` and the Colab/Kaggle notebook (M3b), the
  first real local run (needs your hardware or Kaggle), response-language detection (M4).

## M3b
- `scripts/bootstrap_free_gpu.sh` + `notebooks/indicsafe_free_gpu.ipynb`: install deps, install and
  start Ollama, pull the tags named by `indicsafe models --config`, bench, run, zip. `DRY_RUN=1` prints
  the plan. Tested here end to end with scripted models (`SKIP_SYNC=1 SKIP_OLLAMA=1`), including a
  re-run that repeats no work. **Not tested:** the Ollama install, `ollama serve` and `ollama pull`
  steps, or anything on real Kaggle/Colab (no network or GPU in the build sandbox). Expect to adjust.
- Ollama is installed with its official `curl | sh` installer; the script also installs `zstd`
  with apt if missing (the installer needs it on some versions; unverified).
- Colab sessions lose local disk when they die: set `OUT` to a Drive folder to resume across sessions.
  Kaggle keeps `/kaggle/working` as notebook output.
- Responses can contain harmful-compliance text; the notebook warns to keep outputs private.
  Redacted public exports are planned for M6 (`report`).
