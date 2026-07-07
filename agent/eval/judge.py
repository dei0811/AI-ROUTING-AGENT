"""Local judge for dev-time self-eval. NEVER in the scored path.

Two modes:
- heuristic_judge: deterministic, offline, zero tokens. Category-aware
  matching against the dev set's `expected` field. Used whenever no
  judge client is available (mock development).
- llm_judge: terse PASS/FAIL call through a provided client, for when
  a real Fireworks key is plugged in. Falls back to the heuristic on
  any error.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You judge answers. Given TASK, EXPECTED and ANSWER, reply PASS if "
    "the answer satisfies the task and matches the expected intent, "
    "else FAIL. One word only."
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _contains(haystack: str, needle: str) -> bool:
    """Case-insensitive containment; short needles must match whole words
    (so expected "no" doesn't pass on an answer containing "north")."""
    haystack = haystack.lower()
    needle = needle.lower()
    if len(needle) <= 4 and needle.isalnum():
        return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None
    return needle in haystack


def heuristic_judge(category: str, answer: str, expected: str) -> bool:
    """Deterministic pass/fail for a dev-set answer.

    Rules per category:
    - math: some number in the answer equals the expected number.
    - ner: expected is a JSON list of entity strings; all must appear.
    - everything else: expected phrase appears in the answer.
    """
    answer = (answer or "").strip()
    expected = (expected or "").strip()
    if not answer or not expected:
        return False

    if category == "math":
        try:
            target = float(expected)
        except ValueError:
            return _contains(answer, expected)
        return any(float(n) == target for n in _NUMBER_RE.findall(answer))

    if category == "ner":
        try:
            entities = json.loads(expected)
        except ValueError:
            entities = [expected]
        return all(_contains(answer, str(entity)) for entity in entities)

    return _contains(answer, expected)


def llm_judge(client, model: str, task_prompt: str, answer: str,
              expected: str) -> bool:
    """PASS/FAIL judgement via an LLM (dev-time only, tiny max_tokens)."""
    user = f"TASK: {task_prompt}\nEXPECTED: {expected}\nANSWER: {answer}"
    verdict = client.complete(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=4,
    )
    return "pass" in (verdict or "").lower()


def judge(category: str, task_prompt: str, answer: str, expected: str,
          client=None, model: str = None) -> bool:
    """Judge one answer. Uses the LLM when a client is given, otherwise
    (or on LLM failure) the deterministic heuristic."""
    if client is not None and model:
        try:
            return llm_judge(client, model, task_prompt, answer, expected)
        except Exception:
            logger.exception("LLM judge failed; falling back to heuristic")
    return heuristic_judge(category, answer, expected)
