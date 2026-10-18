"""Dataset validation. Collects every problem, then fails loudly."""

from __future__ import annotations

import re
from collections import Counter

from indicsafe.languages import LANGUAGES, script_errors
from indicsafe.schema import Item

# Backstop only, not a guarantee: rejects obviously out-of-scope severity. Human review of
# every new harmful item is still required (see CLAUDE.md safety rules).
OUT_OF_SCOPE = re.compile(
    r"\b(bomb|explosive|detonat\w*|nerve agent|sarin|anthrax|ricin|bioweapon|"
    r"chemical weapon|nuclear weapon|suicide method|kill myself|self[- ]harm method|"
    r"child porn\w*|csam|sexual\w* .{0,20}\b(minor|child)\b|3d[- ]print\w* (a )?(gun|firearm))\b",
    re.IGNORECASE,
)


class DatasetInvalid(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} dataset problem(s):\n" + "\n".join(problems))


def check_items(items: list[Item]) -> list[str]:
    problems: list[str] = []
    ids = Counter(i.item_id for i in items)
    problems += [f"duplicate item_id {k!r}" for k, n in ids.items() if n > 1]
    by_id = {i.item_id: i for i in items}

    for item in items:
        tag = f"[{item.item_id}]"
        langs = Counter(t.language for t in item.translations)
        problems += [f"{tag} duplicate translation for {k!r}" for k, n in langs.items() if n > 1]
        if "en" not in langs:
            problems.append(f"{tag} missing English translation (needed as grader reference)")
        for t in item.translations:
            spec = LANGUAGES.get(t.language)
            if spec is None:
                problems.append(f"{tag} unknown language tag {t.language!r}")
                continue
            if t.script != spec.script:
                problems.append(
                    f"{tag} {t.language}: script={t.script!r}, expected {spec.script!r}"
                )
            problems += [f"{tag} {t.language}: {e}" for e in script_errors(t.text, spec)]
            if OUT_OF_SCOPE.search(t.text):
                problems.append(f"{tag} {t.language}: matches out-of-scope severity denylist")

        if item.expected_behaviour == "refuse":
            partner = by_id.get(item.paired_item_id or "")
            if partner is None:
                problems.append(f"{tag} harmful item has no valid paired_item_id")
            else:
                if partner.expected_behaviour != "comply":
                    problems.append(f"{tag} paired item {partner.item_id} is not a benign control")
                if partner.category != item.category:
                    problems.append(f"{tag} paired item {partner.item_id} has a different category")
        if item.paired_item_id:
            partner = by_id.get(item.paired_item_id)
            if partner is None:
                problems.append(f"{tag} paired_item_id {item.paired_item_id!r} does not exist")
            elif partner.paired_item_id != item.item_id:
                problems.append(f"{tag} pairing with {partner.item_id} is not reciprocal")
    return problems


def validate_items(items: list[Item]) -> None:
    problems = check_items(items)
    if problems:
        raise DatasetInvalid(problems)
