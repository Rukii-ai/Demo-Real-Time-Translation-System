import pytest
from typing import List

from app.translation import translator as translator_module


class DummyEngine:
    """Tiny fake engine for unit tests."""

    def translate(self, texts, src_lang, tgt_lang, **kwargs):
        single = isinstance(texts, str)
        if single:
            return {"text": f"[UNIT-MOCK:{tgt_lang}] {texts}", "raw": None}
        return {"text": [f"[UNIT-MOCK:{tgt_lang}] {t}" for t in texts], "raw": None}


def test_translator_with_injected_engine():
    engine = DummyEngine()
    t = translator_module.Translator(engine=engine)
    out = t.translate("hello", src_lang="eng_Latn", tgt_lang="fra_Latn")
    assert "[UNIT-MOCK:fra_Latn]" in out["text"]


def test_translator_creates_and_owns_engine(monkeypatch):
    # Replace the NLLBEngine in the translator module with our DummyEngine
    monkeypatch.setattr(translator_module, "NLLBEngine", DummyEngine)

    t = translator_module.Translator(engine_kwargs={})
    # Translator should report owning the engine when it created it
    assert t._owns_engine is True

    out = t.translate("hi there", src_lang=None, tgt_lang="spa_Latn")
    assert "[UNIT-MOCK:spa_Latn]" in out["text"]

    # close() should not raise when engine implements close or not
    t.close()


def test_translate_list_input_and_context_manager(monkeypatch):
    monkeypatch.setattr(translator_module, "NLLBEngine", DummyEngine)
    with translator_module.Translator(engine_kwargs={}) as t:
        texts: List[str] = ["one", "two"]
        out = t.translate(texts, src_lang="eng_Latn", tgt_lang="deu_Latn")
        assert isinstance(out["text"], list)
        assert len(out["text"]) == 2
        assert out["text"][0].startswith("[UNIT-MOCK:deu_Latn]")
