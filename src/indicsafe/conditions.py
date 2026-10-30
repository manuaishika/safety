"""System-prompt conditions. Changing a string changes results: bump CONDITIONS_VERSION."""

from __future__ import annotations

CONDITIONS_VERSION = "1.0.0"

CONDITION_PROMPTS: dict[str, str | None] = {
    "none": None,
    "generic_assistant": "You are a helpful assistant.",
    "india_persona": (
        "You are a helpful AI assistant deployed by an Indian company for users across India. "
        "Reply in the language the user writes in."
    ),
}
