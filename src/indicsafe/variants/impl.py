"""Variant implementations. Registered with explicit names/versions in registry.py."""

from __future__ import annotations

import random
import re

from indicsafe.variants import lexicon, templates
from indicsafe.variants.base import Transform, map_words

Info = dict[str, object]


def identity(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
    return text, {}


def roleplay(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
    options = templates.ROLEPLAY[language]
    i = rng.randrange(len(options))
    return options[i].format(text=text), {"template": i}


def context_prefix(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
    options = templates.CONTEXT_PREFIX[language]
    i = rng.randrange(len(options))
    return options[i] + text, {"template": i}


def make_register(register: str) -> Transform:
    def fn(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
        prefix, suffix = templates.REGISTER[register][language]
        return prefix + text + suffix, {"register": register}

    return fn


def _match_case(src: str, repl: str) -> str:
    if src[:1].isupper() and repl[:1].isalpha():
        return repl[:1].upper() + repl[1:]
    return repl


def make_code_switch(ratio: float) -> Transform:
    """Swap lexicon words to the other language with probability ``ratio`` per candidate.

    en text gets romanised-Hindi words; hi-Latn and hi text get English words.
    """

    def fn(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
        table = {
            "en": lexicon.EN_TO_ROMAN,
            "hi-Latn": lexicon.ROMAN_TO_EN,
            "hi": lexicon.DEV_TO_EN,
        }[language]
        counts = {"candidates": 0, "swapped": 0}

        def swap(core: str, _initial: bool) -> str | None:
            repl = table.get(core.lower() if language != "hi" else core)
            if repl is None:
                return None
            counts["candidates"] += 1
            # Always draw, so the RNG stream does not depend on ratio.
            if rng.random() < ratio:
                counts["swapped"] += 1
                return _match_case(core, repl)
            return None

        out = map_words(text, swap)
        realised = counts["swapped"] / counts["candidates"] if counts["candidates"] else None
        return out, {**counts, "realised_ratio": realised}

    return fn


# Ordered spelling alternations common in romanised Hindi.
_SUBS = (
    ("aa", "a"),
    ("ee", "i"),
    ("oo", "u"),
    ("ai", "e"),
    ("au", "o"),
    ("ph", "f"),
    ("w", "v"),
    ("v", "w"),
    ("z", "j"),
    ("sh", "s"),
    ("ch", "c"),
)
_VOWELS = "aeiou"


def make_translit_noise(p_sub: float, p_drop: float) -> Transform:
    """Spelling variation and vowel dropping in romanised Hindi (hi-Latn only)."""

    def noisy(word: str, rng: random.Random, counts: dict[str, int]) -> str:
        w = word
        for pat, rep in _SUBS:

            def sub(m: re.Match[str], rep: str = rep) -> str:
                if rng.random() < p_sub:
                    counts["subs"] += 1
                    return rep
                return m.group(0)

            w = re.sub(pat, sub, w)
        if len(w) >= 5:
            chars = list(w)
            for k in range(1, len(chars) - 1):
                if chars[k].lower() in _VOWELS and rng.random() < p_drop:
                    chars[k] = ""
                    counts["drops"] += 1
            w = "".join(chars)
        return w

    def fn(text: str, language: str, rng: random.Random) -> tuple[str, Info]:
        counts = {"subs": 0, "drops": 0}

        def word(core: str, _initial: bool) -> str | None:
            new = noisy(core, rng, counts)
            return new if new and new != core else None

        out = map_words(text, word)
        return out, dict(counts)

    return fn
