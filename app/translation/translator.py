"""
Translator service that uses `NLLBEngine`.

- Accepts an `engine` (dependency injection) or creates/owns a singleton `NLLBEngine`.
- Provides a small convenience API, optional language-detection hooking point,
  and a `close()` that delegates to the engine if this Translator created it.
"""

from typing import Optional, Union, List, Dict, Any
import time

from app.utils.logger import get_logger

from .nllb_engine import NLLBEngine


logger = get_logger(__name__)


class Translator:
    def __init__(
        self,
        engine: Optional[NLLBEngine] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        detect_language: bool = False,
    ):
        """
        - `engine`: if provided, Translator will use this instance (no ownership).
        - otherwise, a single NLLBEngine instance is created (singleton) with `engine_kwargs`.
        - `detect_language`: placeholder flag if you want to implement automatic detection.
        """
        self._owns_engine = False
        if engine is not None:
            self.engine = engine
        else:
            self.engine = NLLBEngine(**(engine_kwargs or {}))
            # We don't strictly need to track ownership because NLLBEngine is a singleton;
            # but tracking helps determine whether to call `close()` on shutdown.
            self._owns_engine = True

        # `Translator` is a small convenience wrapper around the heavy NLLBEngine.
        # It keeps service-level responsibilities (request logging, optional
        # language detection) separate from model lifecycle concerns.

        self.detect_language = detect_language
        logger.info(
            "Translator initialized (owns_engine=%s detect_language=%s)",
            self._owns_engine,
            self.detect_language,
        )

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: Optional[str],
        tgt_lang: str,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Translate `text` from `src_lang` to `tgt_lang`.

        - `src_lang` may be None if you want the model to try autodetection.
        - Returns the engine's response (dict with 'text' and 'raw').
        """
        # Optionally insert language detection here if detect_language is True.
        # e.g. use langdetect or fasttext to set src_lang when None.
        # Log only metadata: count/length and languages, not the raw text.
        count = 1 if isinstance(text, str) else len(text)
        logger.info(
            "Translation request (items=%d src_lang=%s tgt_lang=%s)", count, src_lang, tgt_lang
        )

        start = time.perf_counter()
        try:
            result = self.engine.translate(texts=text, src_lang=src_lang, tgt_lang=tgt_lang, **gen_kwargs)
        except Exception:
            logger.exception("Translation failed in engine")
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Translation completed (items=%d tgt_lang=%s duration_ms=%.2f)", count, tgt_lang, elapsed_ms
        )
        return result

    def close(self):
        """Close the engine if this Translator created/owns it."""
        if self._owns_engine and self.engine is not None:
            try:
                self.engine.close()
            except Exception as exc:
                logger.exception("Error while closing engine: %s", exc)

    # Convenience context manager
    def __enter__(self):
        # Ensure engine loaded (optional)
        if hasattr(self.engine, "_ensure_loaded"):
            self.engine._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# --- Usage examples (do not run in production without proper model/config) ---
# Example 1: Translator creates/owns single engine instance
# translator = Translator(engine_kwargs={"model_name": "facebook/nllb-200-distilled-600M", "device": "cpu"})
# result = translator.translate("Hello world", src_lang="eng_Latn", tgt_lang="fra_Latn")
#
# Example 2: Inject a pre-created engine (explicit DI)
# engine = NLLBEngine(model_name="facebook/nllb-200-distilled-600M", device="cpu")
# translator = Translator(engine=engine)
# result = translator.translate("Hello", src_lang="eng_Latn", tgt_lang="spa_Latn")
#
# Example 3: Context manager with automatic cleanup (translator owns engine)
# with Translator(engine_kwargs={"device": "cpu"}) as t:
#     print(t.translate("Good morning", src_lang="eng_Latn", tgt_lang="ita_Latn")["text"])
