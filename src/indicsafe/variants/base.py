"""Variant interface, seeded RNG, and protected-span handling."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from indicsafe.data.hashing import sha256_hex


class UnsupportedLanguage(Exception):
    pass


def make_rng(seed: int, item_id: str, language: str, variant: str) -> random.Random:
    """Independent RNG per (seed, item, language, variant); order of building never matters."""
    key = sha256_hex(f"{seed}\x1f{item_id}\x1f{language}\x1f{variant}")
    return random.Random(int(key[:16], 16))


# Spans no variant may alter: URLs, emails, handles/hashtags, numbers/amounts, ALLCAPS
# acronyms and tokens containing digits.
_PROTECTED = re.compile(
    r"https?://\S+|www\.\S+"
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"
    r"|[@#]\w+"
    r"|[₹$€£]?[0-9०-९][0-9०-९,./:-]*"
    r"|\b[A-Z]{2,}\b"
    r"|\b\w*[0-9०-९]\w*\b"
)


def protected_spans(text: str) -> list[str]:
    return [m.group(0) for m in _PROTECTED.finditer(text)]


def split_protected(text: str) -> list[tuple[str, bool]]:
    """Split into (segment, is_protected) pieces that concatenate back to ``text``."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _PROTECTED.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _is_punct(c: str) -> bool:
    return unicodedata.category(c)[0] in "PS"


def split_word(chunk: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation); marks/matras stay in the core."""
    i, j = 0, len(chunk)
    while i < j and _is_punct(chunk[i]):
        i += 1
    while j > i and _is_punct(chunk[j - 1]):
        j -= 1
    return chunk[:i], chunk[i:j], chunk[j:]


def map_words(text: str, fn: Callable[[str, bool], str | None]) -> str:
    """Apply ``fn(core, sentence_initial)`` to every unprotected word.

    ``fn`` returns the replacement core or None to keep it. Capitalised Latin words that
    are not sentence-initial are treated as names and never passed to ``fn``.
    """
    out: list[str] = []
    sentence_start = True
    for segment, protected in split_protected(text):
        if protected:
            out.append(segment)
            sentence_start = False
            continue
        for chunk in re.split(r"(\s+)", segment):
            if not chunk or chunk.isspace():
                out.append(chunk)
                continue
            lead, core, trail = split_word(chunk)
            if core:
                is_name = (
                    not sentence_start and core[0].isupper() and core.isascii() and core != "I"
                )
                new = None if is_name else fn(core, sentence_start)
                core = new if new is not None else core
            out.append(lead + core + trail)
            if core:
                sentence_start = bool(trail) and trail[-1] in ".!?।॥"
    return "".join(out)


Transform = Callable[[str, str, random.Random], tuple[str, dict[str, object]]]


@dataclass(frozen=True)
class Variant:
    """A pure, seeded, versioned prompt transform."""

    name: str
    version: str
    description: str
    languages: frozenset[str]
    transform: Transform
    params: dict[str, object] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "languages": sorted(self.languages),
            "params": dict(self.params),
        }

    def apply_with_info(
        self, text: str, language: str, rng: random.Random
    ) -> tuple[str, dict[str, object]]:
        if language not in self.languages:
            raise UnsupportedLanguage(
                f"variant {self.name!r} does not support {language!r} "
                f"(supports {sorted(self.languages)})"
            )
        return self.transform(text, language, rng)

    def apply(self, text: str, language: str, rng: random.Random) -> str:
        return self.apply_with_info(text, language, rng)[0]
