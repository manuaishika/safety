"""Language/script registry and script-range checks.

Language tags are BCP-47. Romanized Hindi ("Hinglish") is ``hi-Latn``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

ScriptKind = Literal["native", "romanized"]

# Share of letters/marks that must fall in the expected script for native-script text.
# Not 100%: Hindi prose legitimately contains Latin acronyms such as UPI, HR, Bluetooth.
NATIVE_MIN_SHARE = 0.5


@dataclass(frozen=True)
class LangSpec:
    tag: str
    name: str
    script: ScriptKind
    ranges: tuple[tuple[int, int], ...]


_LATIN = ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F))
_INDIC_BLOCKS = ((0x0900, 0x0DFF),)

LANGUAGES: dict[str, LangSpec] = {
    s.tag: s
    for s in (
        LangSpec("en", "English", "native", _LATIN),
        LangSpec("hi", "Hindi (Devanagari)", "native", ((0x0900, 0x097F),)),
        LangSpec("hi-Latn", "Hinglish (Roman script)", "romanized", _LATIN),
        LangSpec("mr", "Marathi (Devanagari)", "native", ((0x0900, 0x097F),)),
        LangSpec("bn", "Bengali", "native", ((0x0980, 0x09FF),)),
        LangSpec("ta", "Tamil", "native", ((0x0B80, 0x0BFF),)),
        LangSpec("te", "Telugu", "native", ((0x0C00, 0x0C7F),)),
    )
}


def _in(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _letters(text: str) -> list[int]:
    """Code points of letters and combining marks (matras are marks, not letters)."""
    return [ord(c) for c in text if unicodedata.category(c)[0] in "LM"]


def script_errors(text: str, spec: LangSpec) -> list[str]:
    """Return a list of problems with ``text`` for the given language; empty if fine."""
    cps = _letters(text)
    if not cps:
        return ["text contains no letters"]
    if spec.script == "romanized" or spec.tag == "en":
        bad = [cp for cp in cps if not _in(cp, _LATIN)]
        if bad:
            return [f"{spec.tag} text must be Latin-only; found U+{bad[0]:04X}"]
        if any(_in(cp, _INDIC_BLOCKS) for cp in cps):  # pragma: no cover - subsumed above
            return [f"{spec.tag} text contains Indic-script characters"]
        return []
    share = sum(_in(cp, spec.ranges) for cp in cps) / len(cps)
    if share < NATIVE_MIN_SHARE:
        return [
            f"{spec.tag} text is only {share:.0%} in its native script "
            f"(minimum {NATIVE_MIN_SHARE:.0%}); wrong language/script?"
        ]
    return []
