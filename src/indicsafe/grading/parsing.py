"""Strict parsing of grader output. No regex label guessing: JSON object or failure."""

from __future__ import annotations

import json

from pydantic import ValidationError

from indicsafe.grading.models import GraderJSON


class MalformedGraderOutput(Exception):
    pass


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return t


def parse_grader_output(text: str) -> GraderJSON:
    """Accept: the whole text is one JSON object (optionally in a code fence), or exactly one
    top-level JSON object embedded in surrounding prose. Anything else raises."""
    body = _strip_fence(text)
    if not body:
        raise MalformedGraderOutput("empty output")
    decoder = json.JSONDecoder()
    found: list[dict[str, object]] = []
    i = 0
    while (j := body.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(body, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end
    if not found:
        raise MalformedGraderOutput("no JSON object found")
    if len(found) > 1:
        raise MalformedGraderOutput(f"{len(found)} JSON objects found; ambiguous")
    try:
        return GraderJSON.model_validate(found[0])
    except ValidationError as e:
        raise MalformedGraderOutput(
            f"schema violation: {e.errors()[0]['msg']} at {e.errors()[0]['loc']}"
        ) from e
