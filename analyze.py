"""Reads Inspect logs in ./logs and prints per-language results + the English gap."""
from collections import defaultdict
from inspect_ai.log import list_eval_logs, read_eval_log

LANGS = ["en", "hi", "hinglish"]

for info in list_eval_logs("logs"):
    log = read_eval_log(info)
    if log.status != "success" or not log.samples:
        continue
    model = log.eval.model
    # lang -> expected -> [labels]
    tab = defaultdict(lambda: defaultdict(list))
    by_pair = defaultdict(dict)  # pair_id -> lang -> correct?
    for s in log.samples:
        md, sc = s.metadata, next(iter(s.scores.values()))
        tab[md["language"]][md["expected"]].append(sc.answer)
        by_pair[md["pair_id"]][md["language"]] = (sc.value == "C")

    print(f"\n### {model}")
    print("| language | harmful refused | benign answered (no over-refusal) | unclear |")
    print("|---|---|---|---|")
    rates = {}
    for lang in LANGS:
        h, b = tab[lang]["refuse"], tab[lang]["comply"]
        hr = sum(x == "REFUSE" for x in h) / max(len(h), 1)
        bc = sum(x == "COMPLY" for x in b) / max(len(b), 1)
        un = sum(x == "UNCLEAR" for x in h + b)
        rates[lang] = hr
        print(f"| {lang} | {hr:.0%} ({len(h)}) | {bc:.0%} ({len(b)}) | {un} |")
    for lang in ("hi", "hinglish"):
        print(f"\nRefusal gap vs English, {lang}: {rates[lang]-rates['en']:+.0%}")
        flips = [p for p, d in by_pair.items() if d.get("en") and not d.get(lang, True)]
        print(f"  pairs right in English but wrong in {lang}: {flips}")
