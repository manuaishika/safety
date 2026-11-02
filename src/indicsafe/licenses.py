"""Model and dataset licences, with restrictions flagged.

IMPORTANT: this registry was written from memory, not fetched from the sources. Licences change
between model versions and sizes. Treat every row as a pointer to the page you must read, not as
legal advice, and check the exact model tag you use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LicenseInfo:
    family: str
    license: str
    url: str
    flag: str | None  # why a human must read it before redistributing outputs; None = permissive
    notes: str = ""


TEST_DOUBLE = LicenseInfo("test double", "n/a (in-process, no weights)", "", None)
TEST_PROVIDERS = frozenset({"scripted", "scriptedgrader", "mockllm"})

# (regex on the model name, provider stripped, case-insensitive; first match wins). Word
# boundaries matter: "ollama" is a provider, not the Llama family.
REGISTRY: tuple[tuple[str, LicenseInfo], ...] = (
    (
        r"\bqwen2\.5",
        LicenseInfo(
            "Qwen2.5", "Apache-2.0 for most sizes; some sizes use the Qwen licence",
            "https://huggingface.co/Qwen", "check the licence of the exact size you run",
        ),
    ),
    (
        r"\bllama",
        LicenseInfo(
            "Llama 3.x", "Llama Community License (custom)",
            "https://www.llama.com/llama3_1/license/",
            "custom licence: attribution, acceptable-use policy and a large-company clause",
        ),
    ),
    (
        r"\bgemma",
        LicenseInfo(
            "Gemma", "Gemma Terms of Use (custom)", "https://ai.google.dev/gemma/terms",
            "custom licence with a prohibited-use policy that flows down to derivatives",
        ),
    ),
    (
        r"\baya\b",
        LicenseInfo(
            "Aya", "CC-BY-NC 4.0 (non-commercial) plus an acceptable-use addendum",
            "https://huggingface.co/CohereLabs",
            "non-commercial: fine for research, restricts commercial reuse of the model",
        ),
    ),
    (
        r"\bindictrans2",
        LicenseInfo(
            "IndicTrans2", "MIT (code); check the model card",
            "https://github.com/AI4Bharat/IndicTrans2", None,
            "the model repo ships custom code (trust_remote_code)",
        ),
    ),
)  # fmt: skip


def lookup(model_id: str) -> LicenseInfo | None:
    provider, _, name = model_id.partition("/")
    if provider in TEST_PROVIDERS:
        return TEST_DOUBLE
    for pattern, info in REGISTRY:
        if re.search(pattern, name, re.IGNORECASE):
            return info
    return None


def dataset_license(repo: Path) -> LicenseInfo:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (repo / name).exists():
            return LicenseInfo("IndicSafe items and code", f"see {name}", "", None)
    return LicenseInfo(
        "IndicSafe items and code",
        "none chosen",
        "",
        "no LICENSE file: until you add one the repo is all-rights-reserved, which blocks reuse",
        "translations are AI-drafted and unreviewed",
    )


def report_rows(model_ids: list[str], repo: Path) -> list[tuple[str, LicenseInfo | None]]:
    rows: list[tuple[str, LicenseInfo | None]] = [(m, lookup(m)) for m in model_ids]
    rows.append(("dataset: data/items", dataset_license(repo)))
    return rows
