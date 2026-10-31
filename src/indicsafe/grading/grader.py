"""Call one grader model for one sample: strict JSON, retries with feedback, cached."""

from __future__ import annotations

import asyncio

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
)

from indicsafe.grading.cache import GradeCache, cache_key
from indicsafe.grading.models import GraderConfig, GraderOutcome
from indicsafe.grading.parsing import MalformedGraderOutput, parse_grader_output
from indicsafe.grading.rubric import Rubric

RETRY_NOTE = (
    "Your previous reply was not valid. Reply with only one JSON object in exactly the format "
    "described, with no other text."
)


async def grade_one(
    model: Model,
    model_id: str,
    rubric: Rubric,
    prompt: str,
    cfg: GraderConfig,
    cache: GradeCache,
    sem: asyncio.Semaphore,
) -> GraderOutcome:
    key = cache_key(model_id, rubric.version, rubric.system, prompt, cfg.max_tokens)
    if (hit := cache.get(key)) is not None:
        return hit.model_copy(update={"from_cache": True})
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=rubric.system),
        ChatMessageUser(content=prompt),
    ]
    gen = GenerateConfig(temperature=0.0, max_tokens=cfg.max_tokens)
    error = ""
    for attempt in range(1, cfg.max_retries + 2):
        async with sem:
            out = await model.generate(messages, config=gen)
        try:
            parsed = parse_grader_output(out.completion)
        except MalformedGraderOutput as e:
            error = str(e)
            messages = [
                *messages,
                ChatMessageAssistant(content=out.completion),
                ChatMessageUser(content=RETRY_NOTE),
            ]
            continue
        outcome = GraderOutcome(grader_model=model_id, parsed=parsed, attempts=attempt)
        cache.put(key, outcome)
        return outcome
    outcome = GraderOutcome(
        grader_model=model_id, parsed=None, error=error, attempts=cfg.max_retries + 1
    )
    cache.put(key, outcome)
    return outcome
