"""Schemas for grader output, grader configuration and final per-sample grades."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from indicsafe.config import LOCAL_PROVIDERS, provider_of
from indicsafe.schema import Label, Slug


class GraderJSON(BaseModel):
    """What a grader model must return. Extra keys are ignored; the label must be exact."""

    model_config = ConfigDict(extra="ignore")
    label: Label
    harmful_content_provided: bool
    response_language: str = Field(min_length=1)
    rationale: str = ""

    @field_validator("label", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("rationale")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v.strip()[:400]


class TranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    backend: Literal["indictrans2"] = "indictrans2"
    model: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    trust_remote_code: bool = False  # IndicTrans2 needs it; opt in explicitly


class GraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Slug
    graders: list[str] = Field(min_length=1, max_length=3)
    rubric_version: str = "1.0.0"
    tie_policy: Literal["unclear"] = "unclear"
    max_retries: int = Field(2, ge=0, le=5)
    max_tokens: int = Field(300, gt=0)
    max_connections: int = Field(4, gt=0)
    allow_self_grading: bool = False
    allow_hosted: bool = False
    translation: TranslationConfig = TranslationConfig()

    @model_validator(mode="after")
    def _check(self) -> GraderConfig:
        if len(set(self.graders)) != len(self.graders):
            raise ValueError("duplicate grader models")
        hosted = [g for g in self.graders if provider_of(g) not in LOCAL_PROVIDERS]
        if hosted and not self.allow_hosted:
            raise ValueError(f"hosted graders {hosted} need allow_hosted: true")
        return self


def load_grader_config(path: Path) -> GraderConfig:
    return GraderConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class GraderOutcome(BaseModel):
    """One grader's result for one sample: parsed output, or an explicit failure."""

    model_config = ConfigDict(extra="forbid")
    grader_model: str
    parsed: GraderJSON | None
    error: str | None = None
    attempts: int = 1
    from_cache: bool = False


class FinalGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    subject_model: str
    sample_id: str
    epoch: int
    item_id: str
    category: str
    language: str
    variant: str
    condition: str
    split: str
    expected_behaviour: str
    label: Label
    label_source: Literal["ensemble", "rule:empty", "all_graders_failed"]
    grader_labels: dict[str, str | None]
    harmful_content_provided: bool | None
    disagreement: bool
    parse_failures: int
    lexicon_refusal: bool
    lexicon_matches: list[str]
    response_language_heuristic: str
    response_language_graders: dict[str, str | None]
    answered_in_prompt_language: bool
    translated: bool
    rubric_version: str
    grader_config: str
