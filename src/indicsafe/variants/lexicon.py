"""Small en <-> hi word table for code-switching.

Each entry: (english forms, Devanagari forms, romanised Hindi forms). The first form of each
group is the one emitted when swapping *into* that language. Coverage is deliberately small
and common-word only; the realised swap ratio is reported per sample. AI-written, unreviewed.
"""

from __future__ import annotations

Entry = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]

ENTRIES: tuple[Entry, ...] = (
    (("how",), ("कैसे",), ("kaise", "kese")),
    (("what",), ("क्या",), ("kya",)),
    (("where",), ("कहाँ", "कहां"), ("kahan", "kahaan")),
    (("why",), ("क्यों",), ("kyon", "kyun")),
    (("and",), ("और",), ("aur",)),
    (("or",), ("या",), ("ya",)),
    (("but",), ("लेकिन",), ("lekin",)),
    (("if",), ("अगर",), ("agar",)),
    (("because",), ("क्योंकि",), ("kyunki", "kyonki")),
    (("without",), ("बिना",), ("bina",)),
    (("very",), ("बहुत",), ("bahut", "bohot")),
    (("now",), ("अब",), ("ab",)),
    (("please",), ("कृपया",), ("kripya",)),
    (("today",), ("आज",), ("aaj",)),
    (("tomorrow",), ("कल",), ("kal",)),
    (("help",), ("मदद",), ("madad",)),
    (("money",), ("पैसा", "पैसे"), ("paisa", "paise")),
    (("friend",), ("दोस्त",), ("dost",)),
    (("friends",), ("दोस्तों",), ("doston",)),
    (("family",), ("परिवार",), ("parivar",)),
    (("home", "house"), ("घर",), ("ghar",)),
    (("work",), ("काम",), ("kaam",)),
    (("people",), ("लोग",), ("log",)),
    (("news",), ("ख़बर", "खबर"), ("khabar",)),
    (("fake",), ("नकली", "फ़र्ज़ी"), ("nakli", "farzi")),
    (("exam",), ("परीक्षा",), ("pariksha",)),
    (("letter",), ("पत्र",), ("patra",)),
    (("complaint",), ("शिकायत",), ("shikayat",)),
    (("message", "text"), ("संदेश",), ("sandesh",)),
    (("messages",), ("संदेशों",), ("sandeshon",)),
    (("question",), ("सवाल",), ("sawal", "saval")),
    (("answer",), ("जवाब",), ("jawab",)),
    (("problem",), ("समस्या",), ("samasya",)),
    (("election",), ("चुनाव",), ("chunav",)),
    (("date",), ("तारीख़", "तारीख"), ("tarikh",)),
    (("safe",), ("सुरक्षित",), ("surakshit",)),
    (("good",), ("अच्छा", "अच्छी"), ("achha", "accha", "achhi")),
    (("bad",), ("बुरा",), ("bura",)),
    (("mother",), ("माँ",), ("maa",)),
    (("father",), ("पिता",), ("pita",)),
    (("brother",), ("भाई",), ("bhai",)),
    (("sister",), ("बहन",), ("behen", "bahan")),
)


def _build() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    en_to_roman: dict[str, str] = {}
    dev_to_en: dict[str, str] = {}
    roman_to_en: dict[str, str] = {}
    for en, dev, roman in ENTRIES:
        for w in en:
            en_to_roman[w] = roman[0]
        for w in dev:
            dev_to_en[w] = en[0]
        for w in roman:
            roman_to_en[w] = en[0]
    return en_to_roman, dev_to_en, roman_to_en


EN_TO_ROMAN, DEV_TO_EN, ROMAN_TO_EN = _build()
