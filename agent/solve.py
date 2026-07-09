"""Per-task orchestration: classify, route local-vs-Fireworks, call, validate.

Hybrid local-first router (spec §0/§5): every task is classified for
free, then routed once, up front — no generate-then-verify cascade,
which would stack local (~15-25 s) on top of a paid call and bust the
30 s/task limit.

- LOCAL route: bundled llama.cpp model, 0 Fireworks tokens. Runs
  serially (2 vCPU, one context) from a single queue, scheduled first.
- FIREWORKS route: cheapest sufficient tier from ALLOWED_MODELS, terse
  prompt, capped max_tokens, bounded thread pool.

Time budgets (spec §1):
- Global 10 min: solve_all tracks a global deadline; when the serial
  local queue is at risk of not finishing, remaining local tasks are
  flipped to Fireworks (fast, parallel) — a few tokens beat a TIMEOUT.
- Per task < 30 s: each task gets its own deadline. Local generation
  streams and truncates at the deadline; paid retries/escalations are
  gated on remaining headroom.

Model ids are never hardcoded — ALLOWED_MODELS is ranked into
small/mid/large tiers at runtime and config.json maps categories to
tiers (spec §6). Math goes emit-code -> local execution (free) instead
of paying for (or slowly decoding) chain-of-thought tokens.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify import (
    CODE_DEBUG, CODE_GEN, FACTUAL, LOGIC, MATH, NER,
    SENTIMENT, SUMMARIZATION, UNKNOWN, classify,
)
from code_exec import DEFAULT_TIMEOUT_S, extract_code, run_python
from prompts import CODE_EMIT_SYSTEM_PROMPT, DEFAULT_MAX_TOKENS, build_messages

logger = logging.getLogger(__name__)

LOCAL = "local"
FIREWORKS = "fireworks"

# Default routes per spec §6: local wherever a small CPU model tends to
# clear the gate; Fireworks only for code and anything unclassifiable.
# config.json "category_routes" overrides per category.
DEFAULT_ROUTES = {
    FACTUAL: LOCAL,
    MATH: LOCAL,
    SENTIMENT: LOCAL,
    SUMMARIZATION: LOCAL,
    NER: LOCAL,
    CODE_DEBUG: FIREWORKS,
    LOGIC: LOCAL,
    CODE_GEN: FIREWORKS,
    UNKNOWN: LOCAL,
}

# Categories solved via emit-code -> local execution, per route (§6:
# math default is Local + code exec; logic can be added via config).
DEFAULT_CODE_EXEC_CATEGORIES = (MATH,)        # when routed to Fireworks
DEFAULT_LOCAL_CODE_EXEC_CATEGORIES = (MATH,)  # when routed locally

DEFAULT_CONCURRENCY = 8
# 9 min: leaves the 10-min cap a margin for startup + writing results.
DEFAULT_GLOBAL_BUDGET_S = 540.0
# Fireworks tasks: worst case = task budget + one in-flight call (~12 s) < 30 s.
DEFAULT_TASK_BUDGET_S = 16.0
# Local tasks: CPU decode is slow (~5-10 tok/s), so give the local path
# more of the 30 s cap; generation truncates at the deadline either way.
DEFAULT_LOCAL_TASK_BUDGET_S = 25.0
# Minimum remaining per-task budget to start another paid call.
_CALL_HEADROOM_S = 4.0

# Scheduler flip guard: keep answering locally while there is time for
# (one more local task) + (a parallel Fireworks flush of the rest).
DEFAULT_LOCAL_TASK_EST_S = 20.0
DEFAULT_FLIP_RESERVE_S = 60.0

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


def resolve_route(category: str, config: dict, local_model) -> str:
    """Where a category's tasks go by default: LOCAL or FIREWORKS.

    Without a usable local model everything is FIREWORKS (degraded but
    valid). Unknown route values in config fall back to the default.
    """
    if local_model is None:
        return FIREWORKS
    route = config.get("category_routes", {}).get(
        category, DEFAULT_ROUTES.get(category, FIREWORKS)
    )
    return route if route in (LOCAL, FIREWORKS) else DEFAULT_ROUTES.get(category, FIREWORKS)


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
        retry/escalation; cleaned is always the best local normalization.
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
    """Direct-answer Fireworks call with validation and one strict retry.

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


def _solve_via_code_fireworks(prompt: str, client, model: str, max_tokens: int,
                              timeout_s: float, deadline: float) -> str:
    """Paid emit-code path: model writes a program, we run it for free.

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
    return _exec_emitted(completion, timeout_s, deadline)


def _solve_via_code_local(prompt: str, local_model, max_tokens: int,
                          timeout_s: float, deadline: float) -> str:
    """Free emit-code path: local model writes the program."""
    completion = local_model.generate(
        [
            {"role": "system", "content": CODE_EMIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        deadline=deadline,
    )
    return _exec_emitted(completion, timeout_s, deadline)


def _exec_emitted(completion: str, timeout_s: float, deadline: float) -> str:
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


def _solve_local(task_id: str, category: str, prompt: str, local_model,
                 client, models_by_tier: dict, config: dict,
                 max_tokens: int, deadline: float) -> tuple:
    """LOCAL route: free answer, with a paid escape hatch on malformed.

    Returns:
        (answer, source) where source is "local" or "local->fireworks".
    """
    answer = None
    if category in tuple(config.get(
            "local_code_exec_categories", DEFAULT_LOCAL_CODE_EXEC_CATEGORIES)):
        answer = _solve_via_code_local(
            prompt, local_model, max_tokens,
            timeout_s=config.get("code_exec_timeout_s", DEFAULT_TIMEOUT_S),
            deadline=deadline,
        )
        if answer is not None:
            return answer, LOCAL

    raw = local_model.generate(
        build_messages(category, prompt), max_tokens=max_tokens, deadline=deadline,
    )
    ok, cleaned = clean_answer(category, raw)
    if ok:
        return cleaned, LOCAL

    # Malformed local output. A local retry would cost another slow
    # CPU decode inside the same 30 s window; one cheap Fireworks call
    # is the better spend — but only if configured and time remains.
    if (config.get("escalate_malformed_local", True)
            and client is not None
            and deadline - time.monotonic() >= _CALL_HEADROOM_S):
        logger.info("Task %r: malformed local %s answer; escalating to Fireworks",
                    task_id, category)
        tier = config.get("category_tiers", {}).get(category, "mid")
        escalated = _validated_complete(
            client, models_by_tier[tier], category, prompt, max_tokens, deadline,
        )
        if escalated:
            return escalated, "local->fireworks"

    return cleaned, LOCAL  # best-effort local text beats an empty answer


def _solve_fireworks(category: str, prompt: str, client, models_by_tier: dict,
                     config: dict, max_tokens: int, deadline: float) -> tuple:
    """FIREWORKS route (also the flip target when time runs short)."""
    tier = config.get("category_tiers", {}).get(category, "mid")
    model = models_by_tier[tier]

    answer = None
    made_call = False
    if category in tuple(config.get(
            "code_exec_categories", DEFAULT_CODE_EXEC_CATEGORIES)):
        answer = _solve_via_code_fireworks(
            prompt, client, model, max_tokens,
            timeout_s=config.get("code_exec_timeout_s", DEFAULT_TIMEOUT_S),
            deadline=deadline,
        )
        made_call = True
    if answer is None:
        if made_call and deadline - time.monotonic() < _CALL_HEADROOM_S:
            logger.warning("Out of budget before direct call (%s)", category)
            answer = ""
        else:
            answer = _validated_complete(
                client, model, category, prompt, max_tokens, deadline,
            )
    return answer, FIREWORKS


def solve_task(task: dict, client, local_model, models_by_tier: dict,
               config: dict, route: str = None) -> dict:
    """Solve a single task best-effort within its own time budget; never raises.

    Args:
        route: Force LOCAL or FIREWORKS (the scheduler's flip); default
            resolves from config/DEFAULT_ROUTES.

    Returns:
        {"task_id", "answer", "category", "source"}; answer is "" if
        the prompt is empty or every attempt failed (the id must still
        be covered in the output).
    """
    started = time.monotonic()
    result = _solve_task(task, client, local_model, models_by_tier, config, route)
    if os.environ.get("BENCH_TIMING") == "1":
        # Per-task marker for eval/verify_image.py (<30 s rule).
        print(f"TASK {result['task_id']} {time.monotonic() - started:.2f}",
              flush=True)
    return result


def _solve_task(task: dict, client, local_model, models_by_tier: dict,
                config: dict, route: str = None) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    if not prompt.strip():
        logger.warning("Task %r has an empty prompt; writing empty answer", task_id)
        return {"task_id": task_id, "answer": "", "category": "n/a", "source": "none"}

    category = classify(prompt)
    if route is None:
        route = resolve_route(category, config, local_model)
    if route == LOCAL and local_model is None:
        route = FIREWORKS

    budget_key, default_budget = (
        ("local_task_budget_s", DEFAULT_LOCAL_TASK_BUDGET_S) if route == LOCAL
        else ("task_budget_s", DEFAULT_TASK_BUDGET_S)
    )
    deadline = time.monotonic() + config.get(budget_key, default_budget)
    max_tokens = config.get("category_max_tokens", {}).get(
        category, DEFAULT_MAX_TOKENS.get(category, _FALLBACK_MAX_TOKENS)
    )

    try:
        if route == LOCAL:
            answer, source = _solve_local(
                task_id, category, prompt, local_model, client,
                models_by_tier, config, max_tokens, deadline,
            )
        else:
            answer, source = _solve_fireworks(
                category, prompt, client, models_by_tier, config,
                max_tokens, deadline,
            )
    except Exception:
        logger.exception("Task %r (%s) failed; writing empty answer", task_id, category)
        return {"task_id": task_id, "answer": "", "category": category, "source": "error"}

    return {
        "task_id": task_id,
        "answer": (answer or "").strip(),
        "category": category,
        "source": source,
    }


def solve_all(tasks: list, client, config: dict, local_model=None,
              start_time: float = None) -> list:
    """Solve the batch: local queue serial-first, Fireworks pool parallel.

    Scheduling (spec §5/§7): Fireworks-routed tasks go straight onto a
    bounded thread pool (network-bound, cheap to parallelize). Local-
    routed tasks run serially in this thread — llama.cpp on 2 vCPU is
    CPU-bound, parallelism buys nothing. Before each local task the
    global budget is checked: once there is only time left for the
    Fireworks flush, every remaining local task flips to the pool.

    Args:
        tasks: Validated task dicts.
        client: Fireworks (or mock) client; thread-safe. May be None
            only if every task is local-routed and nothing escalates.
        config: Env + file config.
        local_model: LocalModel/MockLocalModel, or None for all-Fireworks.
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

    local_queue = []
    fireworks_tasks = []
    for task in tasks:
        route = resolve_route(classify(task["prompt"]), config, local_model)
        (local_queue if route == LOCAL else fireworks_tasks).append(task)

    logger.info(
        "Solving %d tasks: %d local (serial), %d fireworks (concurrency=%d), "
        "global budget %.0fs remaining",
        len(tasks), len(local_queue), len(fireworks_tasks), concurrency,
        global_deadline - time.monotonic(),
    )

    results_by_id = {}
    source_counts = {}
    category_counts = {}

    def _record(result: dict) -> None:
        results_by_id[result["task_id"]] = result
        source_counts[result["source"]] = source_counts.get(result["source"], 0) + 1
        category_counts[result["category"]] = (
            category_counts.get(result["category"], 0) + 1
        )

    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = [
        executor.submit(solve_task, task, client, local_model,
                        models_by_tier, config)
        for task in fireworks_tasks
    ]

    # Serial local loop with the time-budget guard. The per-task time
    # estimate starts from config and tracks observed durations (EMA),
    # so the flip decision reflects the real CPU speed of this box.
    local_est_s = float(config.get("local_task_est_s", DEFAULT_LOCAL_TASK_EST_S))
    flip_reserve_s = float(config.get("flip_reserve_s", DEFAULT_FLIP_RESERVE_S))
    flipped = 0
    for index, task in enumerate(local_queue):
        remaining = global_deadline - time.monotonic()
        if remaining < flip_reserve_s + local_est_s:
            overflow = local_queue[index:]
            flipped = len(overflow)
            logger.warning(
                "Only %.0fs left for %d local tasks; flipping them to Fireworks",
                remaining, flipped,
            )
            futures.extend(
                executor.submit(solve_task, t, client, local_model,
                                models_by_tier, config, FIREWORKS)
                for t in overflow
            )
            break
        task_start = time.monotonic()
        _record(solve_task(task, client, local_model, models_by_tier, config, LOCAL))
        local_est_s = 0.5 * local_est_s + 0.5 * (time.monotonic() - task_start)

    completed = len(results_by_id)
    try:
        for future in as_completed(
            futures, timeout=max(0.1, global_deadline - time.monotonic())
        ):
            _record(future.result())  # solve_task never raises
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

    if flipped:
        logger.info("Time-budget guard flipped %d local tasks to Fireworks", flipped)
    logger.info("Answer sources: %s", source_counts)
    logger.info("Category distribution: %s", category_counts)

    # Extra keys (category/source) are for logs and dev eval;
    # write_results keeps only task_id/answer in the output file.
    empty = {"answer": "", "category": "n/a", "source": "unanswered"}
    return [
        {
            "task_id": task["task_id"],
            "answer": results_by_id.get(task["task_id"], empty)["answer"],
            "category": results_by_id.get(task["task_id"], empty)["category"],
            "source": results_by_id.get(task["task_id"], empty)["source"],
        }
        for task in tasks
    ]
