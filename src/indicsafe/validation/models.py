"""Schemas for the validation export key and human annotations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

HUMAN_LABELS = ("REFUSE", "PARTIAL", "COMPLY", "UNCLEAR")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyRow(_Strict):
    row_id: str
    subject_model: str
    run_id: str
    sample_id: str
    epoch: int
    item_id: str
    category: str
    language: str
    expected_behaviour: str
    split: str
    grader_label: str
    grader_labels: dict[str, str | None]
    disagreement: bool
    stratum: str  # "<language>|<grader label>"
    weight: float  # design weight: cell population / cell sample (cell = language, label, category)


class ExportKey(_Strict):
    """PRIVATE: holds the grader labels. Never give this file to annotators."""

    export_id: str
    mode: Literal["sample", "all"]
    n_requested: int | None
    seed: int
    include_test: bool
    grades_dir: str
    grades_sha256: str
    rubric_version: str
    grader_config: str
    n_population: int
    n_excluded_empty: int
    n_excluded_test: int
    unsampled_population_share: float
    strata_population: dict[str, int]
    strata_sampled: dict[str, int]
    warnings: list[str]
    rows: list[KeyRow]


class Annotation(_Strict):
    row_id: str
    annotator_id: str
    label: Literal["REFUSE", "PARTIAL", "COMPLY", "UNCLEAR"]
    harmful_content_provided: bool | None = None
    notes: str = ""
