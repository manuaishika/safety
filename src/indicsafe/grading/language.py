"""Response-language detection: a transparent script-and-marker heuristic (no dependencies).

Limits: Devanagari is split hi/mr by marker words only; Roman-script Hindi is detected by
common function words, so short or code-mixed replies can be misjudged. The grader's own
`response_language` is recorded separately; this heuristic is what feeds
`answered_in_prompt_language`.
"""

from __future__ import annotations

import re
import unicodedata

_RANGES = {
    "dev": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
}
_HI_DEV = {"है", "हैं", "और", "नहीं", "का", "की", "के", "में", "को", "से", "यह", "मैं", "आप", "सकता", "सकती"}
_MR_DEV = {"आहे", "नाही", "आणि", "मी", "तुम्ही", "आम्ही", "शकत", "आपण", "काय", "करू"}
_HINGLISH = {
    "hai", "hain", "nahi", "nahin", "main", "mein", "aap", "aapne", "kya", "kaise", "ka", "ki",
    "ke", "ko", "se", "yeh", "ye", "woh", "kar", "karo", "karna", "sakta", "sakti", "madad",
    "aur", "mera", "meri", "mujhe", "tum", "tumhe", "hum", "tha", "thi", "raha", "rahi", "jo",
    "toh", "bhi", "lekin", "zaroor", "chahiye", "kripya", "maaf", "kijiye", "batao", "likho",
}  # fmt: skip
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_of(cp: int) -> str | None:
    for name, (lo, hi) in _RANGES.items():
        if lo <= cp <= hi:
            return name
    if 0x41 <= cp <= 0x24F and (cp <= 0x5A or cp >= 0x61):
        return "latin"
    return None


def detect_language(text: str) -> str:
    """Return en, hi, hi-Latn, mr, ta, bn, te, mixed or unknown."""
    counts: dict[str, int] = {}
    for ch in text:
        if unicodedata.category(ch)[0] in "LM":
            s = _script_of(ord(ch))
            if s:
                counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    top, n = max(counts.items(), key=lambda kv: kv[1])
    if n / total < 0.8:
        big = [s for s, c in counts.items() if c / total >= 0.2]
        if len(big) > 1:
            return "mixed"
    if top == "dev":
        toks = set(text.split())
        hi, mr = len(toks & _HI_DEV), len(toks & _MR_DEV)
        return "mr" if mr > hi else "hi"
    if top == "latin":
        words = [w.lower() for w in _TOKEN.findall(text)]
        hits = sum(w in _HINGLISH for w in words)
        return "hi-Latn" if hits >= 2 and hits / max(len(words), 1) >= 0.2 else "en"
    return top


def answered_in_prompt_language(prompt_language: str, response_language: str) -> bool:
    return prompt_language == response_language
