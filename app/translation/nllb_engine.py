"""
Minimal NLLB engine wrapper (singleton, lazy-loaded).

- Thread-safe singleton: only one instance per process.
- Lazy model/tokenizer loading on first `translate()` call.
- Supports CPU and CUDA; accepts `device_map` for advanced configs.
- Provides `close()` to free resources and `__enter__/__exit__` for context manager usage.
"""

from typing import Optional, Union, List, Dict, Any
import threading
from app.utils.logger import get_logger

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = get_logger(__name__)


class NLLBEngine:
    _instance: Optional["NLLBEngine"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Return existing singleton if created
        # This ensures only one heavy model object exists per process.
        # Many large models use a lot of memory/VRAM, so the singleton prevents
        # accidental double-loading when multiple services import this class.
        if cls._instance is not None:
            return cls._instance

        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # allow __init__ to run normally for the first instantiation
        return cls._instance

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        use_fast: bool = True,
        device_map: Optional[Union[str, dict]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # __init__ may be called multiple times due to singleton pattern;
        # guard against reinitialization. We set up only light-weight fields here
        # and defer heavyweight model/tokenizer loading to `_ensure_loaded()`.
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = model_name
        self.device = device
        self.use_fast = use_fast
        self.device_map = device_map
        self.model_kwargs = model_kwargs or {}

        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._load_lock = threading.Lock()

        self._initialized = True

    # Public API
    def translate(
        self,
        texts: Union[str, List[str]],
        src_lang: Optional[str],
        tgt_lang: str,
        max_length: int = 512,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """
        Translate text(s) from src_lang -> tgt_lang.

        - `texts` can be a single string or a list of strings.
        - `src_lang` can be None (model may auto-detect); `tgt_lang` should be a
          language code the model/tokenizer understands (e.g. "eng_Latn", "fra_Latn").
        - Returns a dict with keys: 'text' (str or list[str]), 'raw' (model outputs).
        """
        # Ensure the tokenizer and model are loaded before translating.
        # Loading is lazy to avoid slow startup when the module is imported.
        self._ensure_loaded()

        single = isinstance(texts, str)
        inputs = [texts] if single else texts

        # Prepare forced BOS token if tokenizer provides language id map, otherwise prefix.
        forced_bos_id = None
        prefix_texts = None
        try:
            lang_map = getattr(self._tokenizer, "lang_code_to_id", None)
            if lang_map and tgt_lang in lang_map:
                forced_bos_id = lang_map[tgt_lang]
            else:
                # Fallback: prepend explicit target language tag token if reasonable.
                # Many NLLB models accept ">>{lang}<< " prefix; keep this as a fallback.
                prefix_texts = [f">>{tgt_lang}<< {t}" for t in inputs]
        except Exception:
            prefix_texts = [f">>{tgt_lang}<< {t}" for t in inputs]

        to_tokenize = prefix_texts if prefix_texts is not None else inputs

        # Tokenize
        enc = self._tokenizer(
            to_tokenize,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move tensors to the model device
        device = torch.device(self.device)
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs = dict(max_length=max_length, **generate_kwargs)
        if forced_bos_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_id

        with torch.no_grad():
            outputs = self._model.generate(**enc, **gen_kwargs)

        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result_text = decoded[0] if single else decoded
        return {"text": result_text, "raw": outputs}

    def _ensure_loaded(self):
        # Lazy load model/tokenizer on first use
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            logger.info("Loading NLLB tokenizer and model: %s", self.model_name)
            # Load tokenizer and model from Hugging Face.
            # This may download weights if not cached locally, which can be slow
            # and requires a valid HF token for private models.
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast)
            # Allow callers to pass a device_map (e.g., "auto") or rely on `device`
            model_load_kwargs = dict(**self.model_kwargs)
            if self.device_map is not None:
                model_load_kwargs["device_map"] = self.device_map
            else:
                # If device is cpu, load on cpu. For cuda, prefer float16 where possible.
                if "cuda" in str(self.device).lower():
                    model_load_kwargs["torch_dtype"] = getattr(torch, "float16", torch.float32)

            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_load_kwargs)

            # If device_map wasn't used, move model to the requested device.
            if self.device_map is None:
                try:
                    self._model.to(torch.device(self.device))
                except Exception:
                    logger.warning("Failed moving model to device %s; continuing", self.device)

            self._loaded = True
            logger.info("NLLBEngine loaded")

    def close(self):
        """Free model and tokenizer from memory. Safe to call multiple times."""
        with self._load_lock:
            if not self._loaded:
                return
            try:
                del self._model
                del self._tokenizer
            except Exception:
                pass
            self._model = None
            self._tokenizer = None
            self._loaded = False

            # If CUDA, clear cache
            if "cuda" in str(self.device).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.info("NLLBEngine closed and memory freed")

    # Context manager convenience
    def __enter__(self):
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()