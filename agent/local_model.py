"""Bundled local model (llama.cpp, CPU): the zero-token answer path.

Answers produced here cost 0 Fireworks tokens and count fully toward
accuracy (spec §0), so the router prefers this path wherever the local
model clears the gate. Constraints that shape this module (spec §1/§4):

- Grading box is 4 GB RAM / 2 vCPU, CPU-only: one model instance,
  small context (KV cache eats RAM), serial generation.
- < 30 s per task at ~5-10 tok/s: outputs are capped short, and
  generation streams so it can stop early at a deadline instead of
  blocking past the task budget.
- Container ready < 60 s: GGUF loads via mmap in seconds; loading
  happens once at startup.

Set MOCK_LOCAL=1 for an offline stand-in (no weights, no llama_cpp),
mirroring MOCK_FIREWORKS in fireworks_client. If weights or the
llama_cpp import are missing, make_local_model returns None and the
router falls back to all-Fireworks — a few tokens beat a zero score.
"""

import glob
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CTX = 2048
# Clamp on any single local completion. At ~5-10 tok/s on 2 vCPU this
# keeps worst-case generation in the ~16-32 s band; per-category caps
# below this still apply (min of the two wins).
DEFAULT_MAX_TOKENS_CAP = 160

# KV-cache quantization: shrinks the biggest non-weight RAM consumer so
# larger models fit the 4 GB box. Values are llama.cpp GGML type names;
# quantized V requires flash attention (CPU-supported in llama.cpp).
DEFAULT_KV_TYPE = "f16"
_KV_TYPE_NAMES = ("f16", "q8_0", "q4_0")

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


class LocalStats:
    """Thread-safe counters for local generation (dev visibility only —
    local tokens are free and never reported to the harness)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.output_tokens = 0
        self.generation_s = 0.0

    def record(self, output_tokens: int, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.output_tokens += output_tokens
            self.generation_s += seconds

    def summary(self) -> dict:
        with self._lock:
            tok_per_s = (
                self.output_tokens / self.generation_s if self.generation_s else 0.0
            )
            return {
                "calls": self.calls,
                "output_tokens": self.output_tokens,
                "generation_s": round(self.generation_s, 1),
                "tok_per_s": round(tok_per_s, 1),
            }


class LocalModel:
    """llama-cpp-python wrapper: load once, generate short capped output.

    Generation is serialized with a lock — llama.cpp contexts are not
    safe for concurrent generate, and on 2 vCPU parallel decode would
    only thrash the cores anyway. The scheduler in solve.py feeds this
    from a single queue.
    """

    def __init__(self, model_path: str, n_ctx: int = DEFAULT_CTX,
                 n_threads: int = None,
                 max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP,
                 kv_type: str = DEFAULT_KV_TYPE):
        import llama_cpp
        from llama_cpp import Llama  # lazy: mock mode must not need it

        kwargs = {}
        if kv_type != "f16":
            if kv_type not in _KV_TYPE_NAMES:
                raise ValueError(f"kv_type must be one of {_KV_TYPE_NAMES}")
            ggml_type = getattr(llama_cpp, f"GGML_TYPE_{kv_type.upper()}")
            # flash_attn is required for a quantized V cache.
            kwargs.update(type_k=ggml_type, type_v=ggml_type, flash_attn=True)

        started = time.monotonic()
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count(),
            verbose=False,
            **kwargs,
        )
        self._lock = threading.Lock()
        self.max_tokens_cap = max_tokens_cap
        self.stats = LocalStats()
        logger.info(
            "Local model loaded in %.1fs: %s (n_ctx=%d, kv=%s)",
            time.monotonic() - started, model_path, n_ctx, kv_type,
        )

    def generate(self, messages: list, max_tokens: int,
                 deadline: float = None) -> str:
        """One short completion; streams so it can stop at the deadline.

        Args:
            messages: OpenAI-style message list (llama.cpp applies the
                model's chat template).
            max_tokens: Output cap; further clamped by max_tokens_cap.
            deadline: time.monotonic() value after which generation is
                cut off, returning whatever was produced so far. This is
                the 30 s/task guard — better a truncated (likely wrong)
                answer for one task than a stuck batch.

        Returns:
            The (possibly deadline-truncated) completion text.
        """
        capped = max(1, min(max_tokens, self.max_tokens_cap))
        started = time.monotonic()
        pieces = []
        n_tokens = 0

        with self._lock:
            stream = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=capped,
                temperature=0.0,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                piece = delta.get("content")
                if piece:
                    pieces.append(piece)
                    n_tokens += 1
                if deadline is not None and time.monotonic() > deadline:
                    logger.warning(
                        "Local generation hit its deadline after %d tokens; truncating",
                        n_tokens,
                    )
                    break

        self.stats.record(n_tokens, time.monotonic() - started)
        return "".join(pieces).strip()


class MockLocalModel:
    """Offline stand-in with the LocalModel interface (MOCK_LOCAL=1).

    Args:
        responses: Canned answers served round-robin (default one
            generic answer). Dev tests can subclass for smarter mocks.
    """

    def __init__(self, responses: list = None):
        self._responses = list(responses) if responses else ["mock local answer"]
        self._index = 0
        self._lock = threading.Lock()
        self.max_tokens_cap = DEFAULT_MAX_TOKENS_CAP
        self.stats = LocalStats()
        self.call_log = []

    def generate(self, messages: list, max_tokens: int,
                 deadline: float = None) -> str:
        with self._lock:
            answer = self._responses[self._index % len(self._responses)]
            self._index += 1
            self.call_log.append({
                "max_tokens": max_tokens,
                "n_messages": len(messages),
            })
        self.stats.record(max(1, len(answer) // 4), 0.0)
        return answer


def find_model_path(config: dict) -> str:
    """Resolve the GGUF path: env override > config > first models/*.gguf."""
    for candidate in (os.environ.get("LOCAL_MODEL_PATH"),
                      config.get("local_model_path")):
        if candidate:
            return candidate

    bundled = sorted(glob.glob(str(_DEFAULT_MODELS_DIR / "*.gguf")))
    return bundled[0] if bundled else None


def make_local_model(config: dict):
    """Build the local model from config/env; None disables the local path.

    Returns MockLocalModel when MOCK_LOCAL=1. Returns None (with a
    warning) when weights are missing or llama_cpp fails to import/load,
    so the caller can degrade to all-Fireworks instead of crashing.
    """
    if os.environ.get("MOCK_LOCAL") == "1":
        logger.info("MOCK_LOCAL=1 -> using MockLocalModel (no weights needed)")
        return MockLocalModel()

    model_path = find_model_path(config)
    if not model_path or not Path(model_path).exists():
        logger.warning(
            "No local GGUF found (LOCAL_MODEL_PATH/config/models/*.gguf); "
            "local route disabled, everything goes to Fireworks."
        )
        return None

    try:
        return LocalModel(
            model_path=model_path,
            n_ctx=int(config.get("local_ctx", DEFAULT_CTX)),
            n_threads=config.get("local_threads"),
            max_tokens_cap=int(
                config.get("local_max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)
            ),
            kv_type=config.get("local_kv_type", DEFAULT_KV_TYPE),
        )
    except Exception:
        logger.exception(
            "Failed to load local model from %s; local route disabled.",
            model_path,
        )
        return None
