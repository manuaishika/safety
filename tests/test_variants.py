from __future__ import annotations

import random
import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import ITEMS_DIR
from indicsafe.data.builder import BuildError, build_samples
from indicsafe.data.hashing import sample_id
from indicsafe.data.loaders import load_items
from indicsafe.data.splits import FreezeLock
from indicsafe.languages import LANGUAGES, script_errors
from indicsafe.variants import (
    REGISTRY,
    UnsupportedLanguage,
    get_variant,
    make_rng,
    protected_spans,
    variant_names,
)

PROTECTED_BITS = [
    "https://example.com/a?b=1",
    "www.site.in",
    "me@mail.com",
    "UPI",
    "HR",
    "9876543210",
    "₹5,000",
    "@user_1",
    "#fakenews",
    "OTP123",
]
WORDS = {
    "en": [
        "how",
        "can",
        "I",
        "report",
        "fake",
        "message",
        "today",
        "Mumbai",
        "Priya",
        "and",
        "help",
    ],
    "hi": ["मुझे", "आज", "नकली", "संदेश", "और", "शिकायत", "करनी", "है", "दिल्ली"],
    "hi-Latn": [
        "main",
        "kaise",
        "aaj",
        "fake",
        "complaint",
        "aur",
        "karun",
        "Delhi",
        "Priya",
        "hai",
    ],
}


@st.composite
def sentences(draw: st.DrawFn, lang: str) -> str:
    pool = WORDS[lang] + PROTECTED_BITS
    toks = draw(st.lists(st.sampled_from(pool), min_size=1, max_size=14))
    end = draw(st.sampled_from(["", ".", "?", "।"]))
    return " ".join(toks) + end


CASES = [(n, lang) for n in variant_names() for lang in sorted(REGISTRY[n].languages)]


def test_all_versions_are_semver() -> None:
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", v.version) for v in REGISTRY.values())


def test_required_variant_families_present() -> None:
    names = set(variant_names())
    for required in (
        "identity",
        "roleplay",
        "context_prefix",
        "register_formal",
        "register_casual",
        "code_switch_25",
        "translit_noise_light",
    ):
        assert required in names


def test_identity_is_exact() -> None:
    v = get_variant("identity")
    assert v.apply("कोई भी text 123", "hi", random.Random(0)) == "कोई भी text 123"


@pytest.mark.parametrize(("name", "lang"), CASES)
@settings(max_examples=40, deadline=None)
@given(data=st.data(), seed=st.integers(0, 10**6))
def test_deterministic_and_preserves_protected(
    name: str, lang: str, data: st.DataObject, seed: int
) -> None:
    text = data.draw(sentences(lang))
    v = get_variant(name)
    a = v.apply_with_info(text, lang, make_rng(seed, "it", lang, name))
    b = v.apply_with_info(text, lang, make_rng(seed, "it", lang, name))
    assert a == b
    out = a[0]
    assert out.strip()
    for span in protected_spans(text):
        assert span in out, f"{name} corrupted {span!r}: {text!r} -> {out!r}"


@pytest.mark.parametrize(
    "name", ["roleplay", "context_prefix", "register_formal", "register_casual"]
)
@pytest.mark.parametrize("lang", ["en", "hi", "hi-Latn"])
def test_wrappers_keep_request_verbatim(name: str, lang: str) -> None:
    text = WORDS[lang][0] + " test 42"
    out = get_variant(name).apply(text, lang, random.Random(1))
    assert text in out and out != text


def test_code_switch_zero_like_ratio_changes_nothing_and_full_swaps_all() -> None:
    src = "How can I report a fake message today and get help"
    full = get_variant("code_switch_100").apply_with_info(src, "en", random.Random(0))
    assert full[1]["realised_ratio"] == 1.0
    for w in ("fake", "message", "today", "help", "and"):
        assert re.search(rf"\b{w}\b", full[0]) is None
    assert full[0].startswith("Kaise")  # capitalisation preserved


def test_code_switch_ratio_is_roughly_respected() -> None:
    src = " ".join(["fake message today help"] * 200)
    info = get_variant("code_switch_50").apply_with_info(src, "en", random.Random(3))[1]
    assert 0.4 < float(info["realised_ratio"]) < 0.6  # type: ignore[arg-type]


def test_code_switch_does_not_touch_names_or_acronyms() -> None:
    out = get_variant("code_switch_100").apply(
        "Please ask Priya about UPI and Mumbai", "en", random.Random(0)
    )
    assert "Priya" in out and "UPI" in out and "Mumbai" in out and "Kripya" in out


def test_noise_only_for_romanised_and_stays_latin() -> None:
    v = get_variant("translit_noise_heavy")
    with pytest.raises(UnsupportedLanguage):
        v.apply("मुझे", "hi", random.Random(0))
    with pytest.raises(UnsupportedLanguage):
        v.apply("hello", "en", random.Random(0))
    text = "Main kaise pehchanun ki yeh message asli hai ya nakli, aaj shaam tak batao"
    changed = 0
    for s in range(20):
        out = v.apply(text, "hi-Latn", make_rng(s, "x", "hi-Latn", v.name))
        assert script_errors(out, LANGUAGES["hi-Latn"]) == []
        changed += out != text
    assert changed >= 15


def test_seed_and_item_change_the_stream() -> None:
    v = get_variant("translit_noise_heavy")
    text = "Main kaise pehchanun ki yeh message asli hai ya nakli, aaj shaam tak batao"
    outs = {v.apply(text, "hi-Latn", make_rng(s, "x", "hi-Latn", v.name)) for s in range(10)}
    assert len(outs) > 1
    assert make_rng(1, "a", "en", "v").random() != make_rng(1, "b", "en", "v").random()


def test_sample_ids_independent_of_registry_growth() -> None:
    # Ids hash names only; these goldens must never change when variants are added.
    assert sample_id("covert_tracking", "hi", "identity", "none") == "586716e73fbfef1a"
    assert len({sample_id("a", "en", n, "none") for n in variant_names()}) == len(variant_names())


def test_every_real_item_builds_for_every_supported_variant(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "l.json")
    langs = ["en", "hi", "hi-Latn"]
    samples = build_samples(items, langs, lock, variants=variant_names(), skip_unsupported=True)
    expected = sum(len(REGISTRY[n].languages & set(langs)) for n in variant_names()) * len(items)
    assert len(samples) == expected
    assert len({s.sample_id for s in samples}) == len(samples)
    ids_identity = {s.sample_id for s in build_samples(items, langs, lock)}
    assert ids_identity <= {s.sample_id for s in samples}  # adding variants keeps old ids
    again = build_samples(items, langs, lock, variants=variant_names(), skip_unsupported=True)
    assert again == samples


def test_builder_fails_loudly_on_unsupported_pair(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    with pytest.raises(BuildError, match="does not support"):
        build_samples(
            items, ["en"], FreezeLock(tmp_path / "l.json"), variants=["translit_noise_light"]
        )


def test_builder_seed_changes_stochastic_variants(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "l.json")
    a = build_samples(items, ["hi-Latn"], lock, variants=["translit_noise_heavy"], seed=1)
    b = build_samples(items, ["hi-Latn"], lock, variants=["translit_noise_heavy"], seed=2)
    assert [s.text for s in a] != [s.text for s in b]
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert all(s.source_text for s in a)
