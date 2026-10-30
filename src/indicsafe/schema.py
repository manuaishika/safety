"""Pydantic models shared by the whole pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "1"

Expected = Literal["refuse", "comply"]
ScriptKind = Literal["native", "romanized"]
ReviewStatus = Literal["unreviewed", "reviewed", "disputed"]
Split = Literal["dev", "test"]
Label = Literal["REFUSE", "PARTIAL", "COMPLY", "UNCLEAR"]

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$", min_length=1)]
Provenance = Annotated[str, StringConstraints(pattern=r"^(human|mt|llm:.+)$")]
Source = Annotated[str, StringConstraints(pattern=r"^(original|adapted:.+)$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Translation(_Strict):
    language: str  # BCP-47, e.g. "en", "hi", "hi-Latn"
    script: ScriptKind
    text: NonEmpty
    provenance: Provenance
    reviewer_id: str | None = None
    review_status: ReviewStatus = "unreviewed"
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def _review_fields(self) -> Translation:
        reviewed = self.review_status != "unreviewed"
        if reviewed and (not self.reviewer_id or self.reviewed_at is None):
            raise ValueError(
                f"review_status={self.review_status} needs reviewer_id and reviewed_at"
            )
        if not reviewed and (self.reviewer_id or self.reviewed_at):
            raise ValueError("unreviewed translation must not carry reviewer_id/reviewed_at")
        return self


class Item(_Strict):
    item_id: Slug
    category: Slug
    subcategory: str | None = None
    expected_behaviour: Expected
    severity_tier: Literal[1, 2, 3]
    india_specific: bool
    paired_item_id: Slug | None = None
    source: Source = "original"
    notes: str = ""
    translations: list[Translation]


class Sample(_Strict):
    """item x language x variant x system-prompt condition, built deterministically."""

    sample_id: str
    item_id: str
    category: str
    expected_behaviour: Expected
    severity_tier: Literal[1, 2, 3]
    india_specific: bool
    split: Split
    language: str
    script: ScriptKind
    variant: str
    variant_version: str
    variant_info: dict[str, object]
    condition: str
    text: str
    source_text: str
    english_reference: str
    provenance: str
    review_status: ReviewStatus


class GradeRecord(_Strict):
    sample_id: str
    label: Label
    harmful_content_provided: bool
    response_language: str
    rationale: str
    grader_model: str
    rubric_version: str


class RunMeta(_Strict):
    run_id: str
    config: dict[str, object]
    dataset_hash: str
    git_sha: str | None
    git_dirty: bool | None
    package_versions: dict[str, str] = Field(default_factory=dict)
    model_ids: list[str]
    grader_ids: list[str] = Field(default_factory=list)
    rubric_version: str | None = None
    seeds: dict[str, int] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None
