from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from indicsafe.languages import LANGUAGES, script_errors
from indicsafe.schema import Translation


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        Translation(language="en", script="native", text="   ", provenance="human")


def test_bad_provenance_rejected() -> None:
    with pytest.raises(ValidationError):
        Translation(language="en", script="native", text="x", provenance="google")


def test_reviewed_needs_reviewer() -> None:
    with pytest.raises(ValidationError):
        Translation(
            language="en", script="native", text="x", provenance="human", review_status="reviewed"
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Translation(language="en", script="native", text="x", provenance="human", oops=1)  # type: ignore[call-arg]


def test_hindi_with_latin_acronym_ok() -> None:
    assert script_errors("मेरा UPI पिन क्या है?", LANGUAGES["hi"]) == []


def test_hindi_in_latin_rejected() -> None:
    assert script_errors("mera pin kya hai", LANGUAGES["hi"])


def test_hinglish_with_devanagari_rejected() -> None:
    assert script_errors("mera पिन kya hai", LANGUAGES["hi-Latn"])


def test_english_with_devanagari_rejected() -> None:
    assert script_errors("hello नमस्ते", LANGUAGES["en"])


def test_tamil_text_in_tamil_spec_ok_and_in_hindi_spec_rejected() -> None:
    text = "எனது கடவுச்சொல் என்ன"
    assert script_errors(text, LANGUAGES["ta"]) == []
    assert script_errors(text, LANGUAGES["hi"])


def test_no_letters_rejected() -> None:
    assert script_errors("123 ???", LANGUAGES["en"])


@given(st.text(alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x7A), min_size=1))
def test_ascii_text_is_never_valid_native_hindi(s: str) -> None:
    if any(c.isalpha() for c in s):
        assert script_errors(s, LANGUAGES["hi"])
