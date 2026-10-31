"""Optional response translation to English (an extra noise source: always recorded)."""

from __future__ import annotations

from typing import Protocol

# IndicTrans2 language codes. It has no Roman-script Hindi, so hi-Latn responses are not translated.
INDICTRANS2_CODES = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "te": "tel_Telu",
}


class Translator(Protocol):
    id: str

    def can_translate(self, language: str) -> bool: ...

    def translate(self, text: str, language: str) -> str: ...


class IndicTrans2Translator:  # pragma: no cover - needs model downloads and GPU/CPU time
    """AI4Bharat IndicTrans2 (Indic -> English). Not exercised in CI or in the build sandbox.

    Needs: transformers, torch and IndicTransToolkit installed, and `trust_remote_code=True`
    (the model repo ships custom code that will run on your machine: read it first).
    """

    def __init__(self, model: str, trust_remote_code: bool) -> None:
        if not trust_remote_code:
            raise RuntimeError(
                "IndicTrans2 needs trust_remote_code: true in the grader config "
                "(it executes code from the model repository)."
            )
        try:
            import torch
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "translation needs transformers, torch and IndicTransToolkit installed"
            ) from e
        self.id = f"indictrans2:{model}"
        self._torch = torch
        self._proc = IndicProcessor(inference=True)
        self._tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model, trust_remote_code=True)

    def can_translate(self, language: str) -> bool:
        return language in INDICTRANS2_CODES

    def translate(self, text: str, language: str) -> str:
        batch = self._proc.preprocess_batch(
            [text], src_lang=INDICTRANS2_CODES[language], tgt_lang="eng_Latn"
        )
        inputs = self._tok(batch, truncation=True, padding="longest", return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(**inputs, num_beams=5, max_length=256, use_cache=True)
        decoded = self._tok.batch_decode(out, skip_special_tokens=True)
        return str(self._proc.postprocess_batch(decoded, lang="eng_Latn")[0])
