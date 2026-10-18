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
