"""Fireworks AI client (OpenAI-compatible) with mock mode and token accounting.

All scored inference must go through FIREWORKS_BASE_URL with a model
from ALLOWED_MODELS (CLAUDE_AGENT_SPEC.md §1). This module enforces
both: the real client is built from the injected env vars only, and
every call is checked against the allowed-model list before being sent.

Token usage reported by the API (`usage`) is accumulated per client so
dev-time evaluation can compare prompt/model choices by total tokens.

Set MOCK_FIREWORKS=1 to build a MockFireworksClient from make_client(),
which lets the whole pipeline run offline before launch day.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Per-call ceiling; keeps any single request well under the 30 s/task limit,
# leaving room for one retry. Overridable for tuning without code changes.
DEFAULT_TIMEOUT_S = float(os.environ.get("FIREWORKS_TIMEOUT_S", "12"))
RETRY_BACKOFF_S = 0.5


class TokenLog:
    """Thread-safe accumulator of token usage across calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def summary(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }


class _BaseClient:
    """Shared allowed-model guardrail + token accounting."""

    def __init__(self, allowed_models: list):
        if not allowed_models:
            raise ValueError("allowed_models must be a non-empty list of model ids")
        self.allowed_models = list(allowed_models)
        self.tokens = TokenLog()

    def _check_model(self, model: str) -> None:
        if model not in self.allowed_models:
            raise ValueError(
                f"Model {model!r} is not in ALLOWED_MODELS {self.allowed_models}; "
                "calling it would invalidate the submission."
            )

    def complete(self, model, messages, max_tokens, temperature=0.0, retries=1):
        raise NotImplementedError


class FireworksClient(_BaseClient):
    """Real client. Routes every call through the injected base URL."""

    def __init__(self, api_key: str, base_url: str, allowed_models: list,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        super().__init__(allowed_models)
        # Lazy import: keeps startup fast and lets mock mode run without openai.
        from openai import OpenAI

        # max_retries=0: we control retries ourselves to bound per-task time.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
        )

    def complete(self, model: str, messages: list, max_tokens: int,
                 temperature: float = 0.0, retries: int = 1) -> str:
        """Single chat completion with capped output and bounded retries.

        Args:
            model: Model id; must be in ALLOWED_MODELS.
            messages: OpenAI-style message list.
            max_tokens: Hard cap on completion tokens (token budget!).
            temperature: 0 by default for determinism.
            retries: Extra attempts on transient failure (default 1).

        Returns:
            The completion text ("" if the API returned no content).

        Raises:
            ValueError: If the model is not allowed.
            Exception: The last error, if all attempts failed.
        """
        self._check_model(model)

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.tokens.record(
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                    )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except ValueError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fireworks call failed (attempt %d/%d): %s",
                    attempt + 1, retries + 1, exc,
                )
                if attempt < retries:
                    time.sleep(RETRY_BACKOFF_S)

        raise last_error


class MockFireworksClient(_BaseClient):
    """Offline stand-in: canned answers + fake (length-estimated) usage.

    Every call is appended to ``call_log`` (model, max_tokens, message
    count) so dev tests can assert routing decisions.

    Args:
        allowed_models: Same guardrail as the real client.
        responses: Optional list of canned answers, served round-robin.
            Defaults to a single generic answer.
    """

    def __init__(self, allowed_models: list, responses: list = None):
        super().__init__(allowed_models)
        self._responses = list(responses) if responses else ["mock answer"]
        self._call_index = 0
        self._lock = threading.Lock()
        self.call_log = []

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def complete(self, model: str, messages: list, max_tokens: int,
                 temperature: float = 0.0, retries: int = 1) -> str:
        self._check_model(model)

        with self._lock:
            answer = self._responses[self._call_index % len(self._responses)]
            self._call_index += 1
            self.call_log.append({
                "model": model,
                "max_tokens": max_tokens,
                "n_messages": len(messages),
            })

        prompt_text = " ".join(str(m.get("content", "")) for m in messages)
        self.tokens.record(
            self._estimate_tokens(prompt_text),
            min(self._estimate_tokens(answer), max_tokens),
        )
        return answer


def make_client(config: dict):
    """Build the right client from the env-derived config.

    Uses MockFireworksClient when MOCK_FIREWORKS=1 (offline dev),
    FireworksClient otherwise.

    Args:
        config: Dict with api_key, base_url, allowed_models
            (as returned by main.read_env_config).
    """
    if os.environ.get("MOCK_FIREWORKS") == "1":
        logger.info("MOCK_FIREWORKS=1 -> using MockFireworksClient (no real calls)")
        return MockFireworksClient(allowed_models=config["allowed_models"])

    return FireworksClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        allowed_models=config["allowed_models"],
    )
