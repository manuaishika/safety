"""Inspect task definitions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample as InspectSample
from inspect_ai.model import ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

import indicsafe.mock as mock
from indicsafe.conditions import CONDITION_PROMPTS
from indicsafe.config import Generation
from indicsafe.data.builder import build_samples
from indicsafe.data.loaders import load_items
from indicsafe.data.splits import FreezeLock
from indicsafe.schema import Sample

_META_FIELDS = (
    "item_id", "category", "expected_behaviour", "severity_tier", "india_specific", "split",
    "language", "script", "variant", "variant_version", "variant_info", "condition",
    "source_text", "english_reference", "provenance", "review_status",
)  # fmt: skip


@solver
def condition_prompt() -> Solver:
    """Prepend the sample's system-prompt condition (none = no system message)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        text = CONDITION_PROMPTS[state.metadata["condition"]]
        if text:
            state.messages.insert(0, ChatMessageSystem(content=text))
        return await generate(state)

    return solve


def to_inspect_samples(samples: Sequence[Sample]) -> list[InspectSample]:
    return [
        InspectSample(
            id=s.sample_id,
            input=s.text,
            target=s.expected_behaviour,
            metadata={f: getattr(s, f) for f in _META_FIELDS},
        )
        for s in samples
    ]


def make_task(
    samples: Sequence[Sample],
    *,
    name: str,
    model: str | None = None,
    generation: Generation | None = None,
    epochs: int = 1,
    metadata: dict[str, object] | None = None,
) -> Task:
    """Generation-only task: responses are stored raw; grading is a separate step (M4)."""
    mock.register_samples(samples)
    g = generation or Generation()
    return Task(
        dataset=MemoryDataset(to_inspect_samples(samples), name=name),
        solver=[condition_prompt()],
        scorer=None,
        config=GenerateConfig(
            temperature=g.temperature,
            max_tokens=g.max_tokens,
            top_p=g.top_p,
            seed=g.seed,
            max_connections=g.max_connections,
        ),
        epochs=epochs,
        model=model,
        name=name,
        metadata=metadata or {},
    )


@task
def indic_refusal(
    languages: str = "en,hi,hi-Latn",
    variants: str = "identity",
    conditions: str = "none",
    categories: str | None = None,
    split: str = "dev",
    seed: int = 0,
    items_dir: str = "data/items",
    lock_path: str = "data/test_freeze.json",
) -> Task:
    """Refusal-consistency generation task, e.g.
    ``inspect eval src/indicsafe/tasks.py --model ollama/<id> -T languages=en,hi``.
    """
    items = load_items(Path(items_dir))
    lock = FreezeLock(Path(lock_path))
    samples = build_samples(
        items,
        languages.split(","),
        lock,
        variants=variants.split(","),
        conditions=conditions.split(","),
        seed=seed,
        skip_unsupported=True,
    )
    cats = set(categories.split(",")) if categories else None
    samples = [
        s for s in samples if (cats is None or s.category in cats) and split in ("all", s.split)
    ]
    return make_task(samples, name="indic_refusal", metadata={"seed": seed})
