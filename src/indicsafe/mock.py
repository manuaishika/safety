"""Scripted, deterministic Inspect model for CI and demos (no network, no tokenizer download).

Model ids look like ``scripted/<policy>``. A prompt's language and expected behaviour are looked
up from a registry filled when samples are built, so behaviour is exact and reproducible.
Test hooks (env vars): INDICSAFE_SCRIPTED_LOG (append one line per call),
INDICSAFE_SCRIPTED_FAIL_AFTER (raise after N calls in this process),
INDICSAFE_SCRIPTED_DELAY (seconds to sleep per call, to simulate slow local inference).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from indicsafe.data.hashing import sha256_hex
from indicsafe.schema import Sample

# policy -> (P(refuse harmful) per language, P(refuse benign))
POLICIES: dict[str, tuple[dict[str, float], float]] = {
    "strict": ({"en": 1.0, "hi": 1.0, "hi-Latn": 1.0}, 0.0),
    "leaky": ({"en": 1.0, "hi": 0.5, "hi-Latn": 0.25}, 0.0),
    "overrefuse": ({"en": 1.0, "hi": 1.0, "hi-Latn": 1.0}, 1.0),
    "compliant": ({"en": 0.0, "hi": 0.0, "hi-Latn": 0.0}, 0.0),
}
REFUSALS = {
    "en": "I can't help with that request.",
    "hi": "मैं इस अनुरोध में मदद नहीं कर सकता।",
    "hi-Latn": "Main is request mein madad nahi kar sakta.",
}
COMPLIANCES = {
    "en": "Sure, here is what you asked for.",
    "hi": "ज़रूर, आपने जो माँगा वह यह रहा।",
    "hi-Latn": "Zaroor, aapne jo manga woh yeh raha.",
}

_REGISTRY: dict[str, tuple[str, str]] = {}  # sha256(prompt) -> (language, expected)
_calls = 0


def register_samples(samples: Iterable[Sample]) -> None:
    for s in samples:
        _REGISTRY[sha256_hex(s.text)] = (s.language, s.expected_behaviour)


def scripted_refuses(policy: str, prompt: str) -> bool:
    """Pure function of (policy, prompt); tests use it to compute expected tables."""
    language, expected = _REGISTRY[sha256_hex(prompt)]
    p_harm, p_benign = POLICIES[policy]
    p = p_harm[language] if expected == "refuse" else p_benign
    u = int(sha256_hex(f"{policy}:{prompt}")[:12], 16) / 16**12
    return u < p


@modelapi(name="scripted")
class ScriptedModel(ModelAPI):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        if model_name not in POLICIES:
            raise ValueError(f"unknown scripted policy {model_name!r}; known: {sorted(POLICIES)}")
        super().__init__(model_name=model_name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        global _calls
        _calls += 1
        limit = os.environ.get("INDICSAFE_SCRIPTED_FAIL_AFTER")
        if limit and _calls > int(limit):
            raise RuntimeError("scripted failure (INDICSAFE_SCRIPTED_FAIL_AFTER)")
        if delay := os.environ.get("INDICSAFE_SCRIPTED_DELAY"):
            await asyncio.sleep(float(delay))
        prompt = input[-1].text
        if (path := os.environ.get("INDICSAFE_SCRIPTED_LOG")) is not None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(sha256_hex(prompt)[:12] + "\n")
        language = _REGISTRY[sha256_hex(prompt)][0]
        text = (REFUSALS if scripted_refuses(self.model_name, prompt) else COMPLIANCES)[language]
        return ModelOutput.from_content(self.model_name, text)
