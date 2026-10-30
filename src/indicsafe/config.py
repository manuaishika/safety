"""Run configuration (YAML) and run ids."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from indicsafe.conditions import CONDITION_PROMPTS
from indicsafe.data.hashing import canonical_json, sha256_hex
from indicsafe.languages import LANGUAGES
from indicsafe.schema import Slug
from indicsafe.variants import variant_names

# Providers that run on the researcher's own hardware (or in-process test doubles).
LOCAL_PROVIDERS = frozenset({"ollama", "hf", "vllm", "scripted", "mockllm"})


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.0
    max_tokens: int = Field(512, gt=0)
    top_p: float | None = None
    seed: int | None = 0
    max_connections: int = Field(4, gt=0)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Slug
    models: list[str] = Field(min_length=1)  # Inspect model strings; ids live only in configs
    languages: list[str] = Field(min_length=1)
    variants: list[str] = ["identity"]
    conditions: list[str] = ["none"]
    categories: list[str] | None = None
    split: Literal["dev", "test", "all"] = "dev"
    seed: int = 0
    epochs: int = Field(1, ge=1)  # k independent samples per prompt (needs temperature > 0)
    skip_unsupported: bool = True  # skipped cells are recorded in RunMeta, never silent
    generation: Generation = Generation()
    allow_hosted: bool = False  # free hosted tiers: off by default, never needed for CI
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        for lang in self.languages:
            if lang not in LANGUAGES:
                raise ValueError(f"unknown language {lang!r}")
        known = set(variant_names())
        for v in self.variants:
            if v not in known:
                raise ValueError(f"unknown variant {v!r}")
        for c in self.conditions:
            if c not in CONDITION_PROMPTS:
                raise ValueError(f"unknown condition {c!r}")
        if self.epochs > 1 and self.generation.temperature <= 0:
            raise ValueError("epochs > 1 needs generation.temperature > 0")
        hosted = [m for m in self.models if provider_of(m) not in LOCAL_PROVIDERS]
        if hosted and not self.allow_hosted:
            raise ValueError(
                f"hosted providers {hosted} need allow_hosted: true (local models only by default)"
            )
        return self


def provider_of(model: str) -> str:
    return model.split("/", 1)[0]


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def run_id(config: RunConfig, dataset_hash: str) -> str:
    """Deterministic: same config + dataset => same id => same log dir => resumable."""
    h = sha256_hex(canonical_json([config.model_dump(mode="json"), dataset_hash]))[:10]
    return f"{config.name}-{h}"
