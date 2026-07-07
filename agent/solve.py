"""Per-task orchestration: classify, pick model+template, call, validate.

Time budgets (spec §1):
- Global: the whole batch must finish well under 10 minutes. solve_all
  runs tasks on a bounded thread pool and stops dispatching results
  once the global budget is spent; unfinished tasks get empty answers
  so every task_id is still covered.
- Per task: < 30 s. Each task gets its own deadline; every paid call
  is gated on remaining budget (headroom >= the per-call timeout), and
  all retries happen at this level with retries=0 on the transport, so
  the worst chain stays bounded: task_budget + one call timeout < 30 s.

Model ids are never hardcoded — ALLOWED_MODELS is ranked into
small/mid/large tiers at runtime and config.json maps categories to
tiers (spec §6). Math goes emit-code -> local execution (free) instead
of paying for chain-of-thought tokens.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify import CODE_DEBUG, CODE_GEN, MATH, NER, SENTIMENT, classify
from code_exec import DEFAULT_TIMEOUT_S, extract_code, run_python
from prompts import CODE_EMIT_SYSTEM_PROMPT, DEFAULT_MAX_TOKENS, build_messages

logger = logging.getLogger(__name__)

# Categories solved via emit-code -> local execution. Config can extend
# this (e.g. add "logic") during launch-day tuning.
DEFAULT_CODE_EXEC_CATEGORIES = (MATH,)

DEFAULT_CONCURRENCY = 8
# 9 min: leaves the 10-min cap a margin for startup + writing results.
DEFAULT_GLOBAL_BUDGET_S = 540.0
# Worst case per task = task budget + one in-flight call (~12 s) < 30 s.
DEFAULT_TASK_BUDGET_S = 16.0
# Minimum remaining budget to start another paid call.
_CALL_HEADROOM_S = 4.0

# Parameter-count hint in model ids, e.g. "llama-v3p1-8b", "qwen2-72B".
_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:\b|[-_])", re.IGNORECASE)
_SMALL_HINTS = ("mini", "tiny", "small", "lite", "nano")
_LARGE_HINTS = ("large", "-xl", "max")

_FALLBACK_MAX_TOKENS = 256

_SENTIMENT_LABELS = ("positive", "negative", "neutral")
_STRICT_SUFFIX = " Strictly output the required format only, nothing else."
_JSON_START_RE = re.compile(r"[\[{]")


def _model_size_score(model_id: str) -> float:
    """Heuristic size of a model, for ranking cheapest -> strongest."""
    lowered = model_id.lower()

    match = _PARAM_COUNT_RE.search(lowered)
    if match:
        return float(match.group(1))
    if any(hint in lowered for hint in _SMALL_HINTS):
        return 1.0
    if any(hint in lowered for hint in _LARGE_HINTS):
        return 1000.0
    return 100.0  # unknown -> treat as mid-sized


def resolve_model_tiers(allowed_models: list) -> dict:
    """Rank ALLOWED_MODELS by estimated size into small/mid/large tiers.

    With fewer than three models, tiers collapse onto what exists
    (one model -> all tiers are that model).

    Returns:
        {"small": id, "mid": id, "large": id}
    """
    ranked = sorted(allowed_models, key=_model_size_score)
    tiers = {
        "small": ranked[0],
        "mid": ranked[len(ranked) // 2],
        "large": ranked[-1],
    }
    logger.info("Model tiers: %s", tiers)
    return tiers


def _extract_json(text: str) -> str:
    """Find and re-serialize the first valid JSON value in ``text``.

    Returns compact JSON, or None if nothing parses. Local and free —
    fixing format here beats paying for a retry call.
    """
    candidate = extract_code(text)  # also strips ```json fences
    decoder = json.JSONDecoder()
    for match in _JSON_START_RE.finditer(candidate):
        try:
            value, _ = decoder.raw_decode(candidate, match.start())
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return None


def clean_answer(category: str, text: str) -> tuple:
    """Locally normalize an answer and judge whether it is well-formed.

    Args:
        category: Task category (drives the expected shape).
        text: Raw completion text.

    Returns:
        (ok, cleaned): ok=False means the caller should spend one cheap
        retry; cleaned is always the best local normalization so far.
    """
    text = (text or "").strip()

    if category == SENTIMENT:
        lowered = text.lower()
        found = [label for label in _SENTIMENT_LABELS if label in lowered]
        if len(found) == 1:
            return True, found[0]
        return False, text

    if category == NER:
        extracted = _extract_json(text)
        if extracted is not None:
            return True, extracted
        return False, text

    if category in (CODE_DEBUG, CODE_GEN):
        code = extract_code(text)
        return bool(code), code

    return bool(text), text


def _try_complete(client, model: str, messages: list, max_tokens: int) -> str:
    """One transport attempt (retries=0); None instead of raising."""
    try:
        return client.complete(
            model=model, messages=messages, max_tokens=max_tokens, retries=0,
        )
    except ValueError:
        raise  # disallowed model: a bug, never swallow it
    except Exception as exc:
        logger.warning("Call failed: %s", exc)
        return None


def _validated_complete(client, model: str, category: str, prompt: str,
                        max_tokens: int, deadline: float) -> str:
    """Direct-answer call with local validation and one strict retry.

    temperature=0 would reproduce the same bad output, so the retry
    tightens the system prompt instead of just re-asking. The retry is
    skipped when the task budget lacks headroom for another call.
    """
    messages = build_messages(category, prompt)
    answer = _try_complete(client, model, messages, max_tokens)
    ok, cleaned = clean_answer(category, answer or "")
    if ok:
        return cleaned

    if deadline - time.monotonic() < _CALL_HEADROOM_S:
        logger.warning("No budget for a retry (%s); best-effort answer", category)
        return cleaned

    logger.info("Malformed %s answer; retrying once with strict prompt", category)
    messages[0]["content"] += _STRICT_SUFFIX
    retry_answer = _try_complete(client, model, messages, max_tokens)
    ok, retry_cleaned = clean_answer(category, retry_answer or "")
    if ok:
        return retry_cleaned

    # Both malformed: ship the best non-empty text rather than nothing.
    return retry_cleaned or cleaned


def _solve_via_code(prompt: str, client, model: str, max_tokens: int,
                    timeout_s: float, deadline: float) -> str:
    """Emit-code path: ask for a program, run it locally, use its stdout.

    Returns:
        The program's output, or None if emission/execution failed or
        the budget ran out (caller falls back to the direct path).
    """
    completion = _try_complete(
        client, model,
        [
            {"role": "system", "content": CODE_EMIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens,
    )
    if completion is None:
        return None

    remaining = deadline - time.monotonic()
    if remaining < 1.0:
        logger.warning("No budget to execute emitted code")
        return None

    code = extract_code(completion)
    ok, output = run_python(code, timeout_s=min(timeout_s, remaining))
    if ok:
        return output

    logger.warning("Emitted code failed (%s); falling back to direct answer", output)
    return None


def solve_task(task: dict, client, models_by_tier: dict, config: dict) -> dict:
    """Solve a single task best-effort within its own time budget; never raises.

    Returns:
        {"task_id": ..., "answer": ..., "category": ...}; answer is ""
        if the prompt is empty or every attempt failed (the id must
        still be covered in the output).
    """
    task_id = task["task_id"]
    prompt = task["prompt"]
    deadline = time.monotonic() + config.get("task_budget_s", DEFAULT_TASK_BUDGET_S)

    if not prompt.strip():
        logger.warning("Task %r has an empty prompt; writing empty answer", task_id)
        return {"task_id": task_id, "answer": "", "category": "n/a"}

    category = classify(prompt)
    tier = config.get("category_tiers", {}).get(category, "mid")
    model = models_by_tier[tier]
    max_tokens = config.get("category_max_tokens", {}).get(
        category, DEFAULT_MAX_TOKENS.get(category, _FALLBACK_MAX_TOKENS)
    )
    code_exec_categories = config.get(
        "code_exec_categories", DEFAULT_CODE_EXEC_CATEGORIES
    )

    try:
        # The first call is never budget-gated: every task must make at
        # least one Fireworks call (spec §11.5 default). Only follow-up
        # calls are gated on remaining headroom.
        answer = None
        made_call = False
        if category in code_exec_categories:
            answer = _solve_via_code(
                prompt, client, model, max_tokens,
                timeout_s=config.get("code_exec_timeout_s", DEFAULT_TIMEOUT_S),
                deadline=deadline,
            )
            made_call = True
        if answer is None:
            if made_call and deadline - time.monotonic() < _CALL_HEADROOM_S:
                logger.warning("Task %r out of budget before direct call", task_id)
                answer = ""
            else:
                answer = _validated_complete(
                    client, model, category, prompt, max_tokens, deadline,
                )
    except Exception:
        logger.exception("Task %r (%s) failed; writing empty answer", task_id, category)
        return {"task_id": task_id, "answer": "", "category": category}

    return {"task_id": task_id, "answer": (answer or "").strip(), "category": category}


def solve_all(tasks: list, client, config: dict, start_time: float = None) -> list:
    """Solve the whole batch with bounded concurrency and a global deadline.

    Args:
        tasks: Validated task dicts.
        client: Fireworks (or mock) client; both are thread-safe.
        config: Env + file config.
        start_time: time.monotonic() at process start, so startup time
            counts against the global budget. Defaults to "now".

    Returns:
        One result entry per task, in input order. Tasks that did not
        finish before the global deadline get empty answers.
    """
    start = start_time if start_time is not None else time.monotonic()
    global_deadline = start + config.get("global_budget_s", DEFAULT_GLOBAL_BUDGET_S)
    concurrency = max(1, int(config.get("concurrency", DEFAULT_CONCURRENCY)))

    models_by_tier = resolve_model_tiers(config["allowed_models"])
    logger.info(
        "Solving %d tasks (concurrency=%d, global budget %.0fs remaining)",
        len(tasks), concurrency, global_deadline - time.monotonic(),
    )

    results_by_id = {}
    category_counts = {}
    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = {
        executor.submit(solve_task, task, client, models_by_tier, config): task
        for task in tasks
    }
    completed = 0
    try:
        for future in as_completed(
            futures, timeout=max(0.1, global_deadline - time.monotonic())
        ):
            result = future.result()  # solve_task never raises
            results_by_id[result["task_id"]] = result["answer"]
            category_counts[result["category"]] = (
                category_counts.get(result["category"], 0) + 1
            )
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                logger.info("Progress: %d/%d tasks", completed, len(tasks))
    except TimeoutError:
        logger.error(
            "Global deadline hit with %d/%d tasks done; writing empty answers "
            "for the rest", completed, len(tasks),
        )
    finally:
        # Don't block on threads stuck in a slow call; results are final.
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info("Category distribution: %s", category_counts)
    return [
        {"task_id": task["task_id"], "answer": results_by_id.get(task["task_id"], "")}
        for task in tasks
    ]
