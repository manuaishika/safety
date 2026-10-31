"""Versioned rubric files and grader prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import yaml

from indicsafe.data.hashing import sha256_hex

DEFAULT_RUBRIC_VERSION = "1.0.0"


@dataclass(frozen=True)
class Rubric:
    version: str
    system: str
    template: str
    translation_block: str
    sha256: str

    def user_prompt(
        self, request: str, request_english: str, response: str, translation: str | None
    ) -> str:
        block = (
            self.translation_block.format(translation=translation)
            if translation is not None
            else ""
        )
        return self.template.format(
            request=request,
            request_english=request_english,
            response=response,
            translation_block=block,
        )


def load_rubric(version: str = DEFAULT_RUBRIC_VERSION) -> Rubric:
    name = f"{version}.yaml"
    try:
        raw = (resources.files("indicsafe.grading") / "rubrics" / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"unknown rubric version {version!r}") from None
    data = yaml.safe_load(raw)
    if data["version"] != version:
        raise ValueError(f"rubric file {name} declares version {data['version']!r}")
    return Rubric(
        version, data["system"], data["template"], data["translation_block"], sha256_hex(raw)
    )
