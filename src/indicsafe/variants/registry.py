"""Variant registry. Adding a variant never changes existing sample_ids (ids use names only).

Any change to a variant's behaviour or wrapper strings must bump its ``version``.
"""

from __future__ import annotations

from indicsafe.variants import impl
from indicsafe.variants.base import UnsupportedLanguage, Variant

_ALL = frozenset({"en", "hi", "hi-Latn"})
_INDIC = frozenset({"hi", "hi-Latn"})

REGISTRY: dict[str, Variant] = {}


def _register(v: Variant) -> None:
    if v.name in REGISTRY:
        raise ValueError(f"duplicate variant {v.name!r}")
    REGISTRY[v.name] = v


_register(Variant("identity", "1.0.0", "Unmodified text.", _ALL, impl.identity))
_register(
    Variant(
        "roleplay", "1.0.0", "Persona/roleplay wrapper around the request.", _ALL, impl.roleplay
    )
)
_register(
    Variant(
        "context_prefix",
        "1.0.0",
        "Benign-context prefix (lawyer / journalist / researcher).",
        _ALL,
        impl.context_prefix,
    )
)
for _reg in ("formal", "casual"):
    _register(
        Variant(
            f"register_{_reg}",
            "1.0.0",
            f"{_reg.capitalize()} register via greeting/closing wrapper (core text unchanged).",
            _ALL,
            impl.make_register(_reg),
            {"register": _reg},
        )
    )
for _pct in (25, 50, 100):
    _register(
        Variant(
            f"code_switch_{_pct}",
            "1.0.0",
            f"Mid-sentence en<->hi swaps of lexicon words, ratio {_pct}%.",
            _ALL,
            impl.make_code_switch(_pct / 100),
            {"ratio": _pct / 100},
        )
    )
for _name, _ps, _pd in (("light", 0.15, 0.05), ("heavy", 0.4, 0.2)):
    _register(
        Variant(
            f"translit_noise_{_name}",
            "1.0.0",
            f"Romanised-Hindi spelling variation and vowel dropping ({_name}).",
            frozenset({"hi-Latn"}),
            impl.make_translit_noise(_ps, _pd),
            {"p_sub": _ps, "p_drop": _pd},
        )
    )


def get_variant(name: str) -> Variant:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(REGISTRY)}") from None


def variant_names() -> list[str]:
    return sorted(REGISTRY)


__all__ = ["REGISTRY", "UnsupportedLanguage", "Variant", "get_variant", "variant_names"]
