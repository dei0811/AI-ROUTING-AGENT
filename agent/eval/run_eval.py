"""Dev-time eval: accuracy gate + token cost on the local dev set.

Runs the real solving pipeline (classify -> route local/Fireworks ->
prompts -> code exec -> validation) over eval/dev_set.jsonl, judges
every answer, and reports per-category pass-rate, the local/Fireworks
answer split, and total Fireworks token usage — the axes that decide
the leaderboard (gate first, fewest tokens second; local answers are
free).

Usage (from agent/):
    python eval/run_eval.py            # real Fireworks + bundled GGUF (env vars required)
    MOCK_FIREWORKS=1 MOCK_LOCAL=1 python eval/run_eval.py             # perfect mocks
    MOCK_FIREWORKS=1 MOCK_LOCAL=1 python eval/run_eval.py --garbage   # worthless mocks

The two mock modes validate the harness itself: perfect models must
score ~100%, useless ones ~0%. With a real key, the LLM judge is used;
in mock mode, the deterministic heuristic judge.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fireworks_client import MockFireworksClient, make_client  # noqa: E402
from io_utils import load_config  # noqa: E402
from judge import judge  # noqa: E402
from local_model import MockLocalModel, make_local_model  # noqa: E402
from solve import solve_all  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

DEV_SET_PATH = Path(__file__).resolve().parent / "dev_set.jsonl"


def load_dev_set(path: Path = DEV_SET_PATH) -> list:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _perfect_answer(task: dict, emit_code: bool) -> str:
    """What a well-behaved model would output for a dev task.

    Args:
        emit_code: True when the call asked for a program (the emit-code
            system prompt), so math answers come back as Python.
    """
    category, expected = task["category"], task["expected"]
    if category == "math" and emit_code:
        return f"```python\nprint({json.dumps(expected)})\n```"
    if category == "math":
        return expected
    if category == "ner":
        return json.dumps({"entities": json.loads(expected)})
    if category == "summarization":
        return f"The text is mainly about {expected}."
    if category in ("code_debug", "code_gen"):
        return f"```python\n{expected}\n```"
    return expected


def _is_emit_code_call(messages: list) -> bool:
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    return "program" in system


def _user_text(messages: list) -> str:
    return next((m["content"] for m in messages if m.get("role") == "user"), "")


class PerfectMock(MockFireworksClient):
    """Fireworks mock that always answers correctly.

    Keyed by the task prompt found in the user message, so it is safe
    under concurrency (unlike round-robin canned responses).
    """

    def __init__(self, allowed_models: list, dev_tasks: list):
        super().__init__(allowed_models)
        self._by_prompt = {task["prompt"]: task for task in dev_tasks}

    def complete(self, model, messages, max_tokens, temperature=0.0, retries=1):
        # Thread-safe by construction (no shared round-robin state):
        # the answer is derived from the message content alone.
        self._check_model(model)
        task = self._by_prompt.get(_user_text(messages))
        answer = (
            _perfect_answer(task, _is_emit_code_call(messages))
            if task else "unknown task"
        )

        with self._lock:
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


class PerfectLocalMock(MockLocalModel):
    """Local mock that always answers correctly (same keying as PerfectMock)."""

    def __init__(self, dev_tasks: list):
        super().__init__()
        self._by_prompt = {task["prompt"]: task for task in dev_tasks}

    def generate(self, messages, max_tokens, deadline=None):
        task = self._by_prompt.get(_user_text(messages))
        answer = (
            _perfect_answer(task, _is_emit_code_call(messages))
            if task else "unknown task"
        )
        with self._lock:
            self.call_log.append({
                "max_tokens": max_tokens,
                "n_messages": len(messages),
            })
        self.stats.record(max(1, len(answer) // 4), 0.0)
        return answer


def build_clients(config: dict, garbage: bool, dev_tasks: list) -> tuple:
    """(fireworks_client, local_model) per the MOCK_* env switches."""
    if os.environ.get("MOCK_FIREWORKS") == "1":
        if garbage:
            client = MockFireworksClient(
                allowed_models=config["allowed_models"],
                responses=["I don't know."],
            )
        else:
            client = PerfectMock(config["allowed_models"], dev_tasks)
    else:
        client = make_client(config)

    if os.environ.get("MOCK_LOCAL") == "1":
        if garbage:
            local_model = MockLocalModel(responses=["I don't know."])
        else:
            local_model = PerfectLocalMock(dev_tasks)
    else:
        local_model = make_local_model(config)

    return client, local_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garbage", action="store_true",
                        help="mock useless models (harness self-test)")
    args = parser.parse_args()

    mock_mode = os.environ.get("MOCK_FIREWORKS") == "1"
    if mock_mode:
        os.environ.setdefault("FIREWORKS_API_KEY", "mock")
        os.environ.setdefault("FIREWORKS_BASE_URL", "http://mock")
        os.environ.setdefault("ALLOWED_MODELS", "mock-8b,mock-70b")

    config = {
        "allowed_models": [
            m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()
        ],
        "api_key": os.environ["FIREWORKS_API_KEY"],
        "base_url": os.environ["FIREWORKS_BASE_URL"],
    }
    config.update(load_config())

    dev_tasks = load_dev_set()
    client, local_model = build_clients(config, args.garbage, dev_tasks)

    # Judge client: real key -> LLM judge on the largest model; mock -> heuristic.
    judge_client = None
    judge_model = None
    if not mock_mode:
        judge_client = make_client(config)
        judge_model = sorted(config["allowed_models"])[-1]

    results = solve_all(dev_tasks, client, config, local_model=local_model)
    by_id = {r["task_id"]: r for r in results}

    per_category = {}
    source_counts = {}
    failures = []
    for task in dev_tasks:
        category = task["category"]
        result = by_id.get(task["task_id"], {})
        answer = result.get("answer", "")
        source = result.get("source", "unanswered")
        source_counts[source] = source_counts.get(source, 0) + 1
        passed = judge(category, task["prompt"], answer, task["expected"],
                       client=judge_client, model=judge_model)
        n, p = per_category.get(category, (0, 0))
        per_category[category] = (n + 1, p + (1 if passed else 0))
        if not passed:
            failures.append((task["task_id"], source, answer))

    print(f"\n{'category':<16}{'pass':>6}{'n':>4}{'rate':>8}")
    print("-" * 34)
    total_n = total_p = 0
    for category in sorted(per_category):
        n, p = per_category[category]
        total_n += n
        total_p += p
        print(f"{category:<16}{p:>6}{n:>4}{p / n:>8.0%}")
    print("-" * 34)
    print(f"{'TOTAL':<16}{total_p:>6}{total_n:>4}{total_p / total_n:>8.0%}")

    print(f"\nanswer sources: {source_counts}")
    print(f"fireworks tokens (scored): {client.tokens.summary()}")
    if local_model is not None:
        print(f"local usage (free): {local_model.stats.summary()}")
    if judge_client is not None:
        print(f"judge tokens (dev only): {judge_client.tokens.summary()}")

    if failures:
        print(f"\nfailed ({len(failures)}):")
        for task_id, source, answer in failures:
            print(f"  {task_id} [{source}]: {answer[:80]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
