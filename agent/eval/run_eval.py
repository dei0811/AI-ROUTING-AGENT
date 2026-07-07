"""Dev-time eval: accuracy gate + token cost on the local dev set.

Runs the real solving pipeline (classify -> prompts -> client -> code
exec -> validation) over eval/dev_set.jsonl, judges every answer, and
reports per-category pass-rate plus total token usage — the two axes
that decide the leaderboard (gate first, fewest tokens second).

Usage (from agent/):
    python eval/run_eval.py            # real Fireworks (env vars required)
    MOCK_FIREWORKS=1 python eval/run_eval.py             # perfect mock
    MOCK_FIREWORKS=1 python eval/run_eval.py --garbage   # worthless mock

The two mock modes exist to validate the harness itself: a perfect
model must score ~100%, a useless one ~0%. With a real key, the LLM
judge is used; in mock mode, the deterministic heuristic judge.
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


class PerfectMock(MockFireworksClient):
    """Simulates a model that always answers correctly.

    Keyed by the task prompt found in the user message, so it is safe
    under concurrency (unlike round-robin canned responses). Math tasks
    get a small program (exercising the local code-exec path), the rest
    get an output shaped like a well-behaved model's.
    """

    def __init__(self, allowed_models: list, dev_tasks: list):
        super().__init__(allowed_models)
        self._by_prompt = {task["prompt"]: task for task in dev_tasks}

    def _answer_for(self, task: dict) -> str:
        category, expected = task["category"], task["expected"]
        if category == "math":
            return f"```python\nprint({json.dumps(expected)})\n```"
        if category == "ner":
            return json.dumps({"entities": json.loads(expected)})
        if category == "summarization":
            return f"The text is mainly about {expected}."
        if category in ("code_debug", "code_gen"):
            return f"```python\n{expected}\n```"
        return expected

    def complete(self, model, messages, max_tokens, temperature=0.0, retries=1):
        # Thread-safe by construction (no shared round-robin state):
        # the answer is derived from the message content alone.
        self._check_model(model)
        user_text = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        task = self._by_prompt.get(user_text)
        answer = self._answer_for(task) if task else "unknown task"

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


def build_solver_client(config: dict, garbage: bool, dev_tasks: list):
    if os.environ.get("MOCK_FIREWORKS") == "1":
        if garbage:
            return MockFireworksClient(
                allowed_models=config["allowed_models"],
                responses=["I don't know."],
            )
        return PerfectMock(config["allowed_models"], dev_tasks)
    return make_client(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garbage", action="store_true",
                        help="mock a useless model (harness self-test)")
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
    client = build_solver_client(config, args.garbage, dev_tasks)

    # Judge client: real key -> LLM judge on the largest model; mock -> heuristic.
    judge_client = None
    judge_model = None
    if not mock_mode:
        judge_client = make_client(config)
        judge_model = sorted(config["allowed_models"])[-1]

    results = solve_all(dev_tasks, client, config)
    answers = {r["task_id"]: r["answer"] for r in results}

    per_category = {}
    failures = []
    for task in dev_tasks:
        category = task["category"]
        answer = answers.get(task["task_id"], "")
        passed = judge(category, task["prompt"], answer, task["expected"],
                       client=judge_client, model=judge_model)
        n, p = per_category.get(category, (0, 0))
        per_category[category] = (n + 1, p + (1 if passed else 0))
        if not passed:
            failures.append((task["task_id"], answer))

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

    print(f"\nsolver tokens: {client.tokens.summary()}")
    if judge_client is not None:
        print(f"judge tokens (dev only): {judge_client.tokens.summary()}")

    if failures:
        print(f"\nfailed ({len(failures)}):")
        for task_id, answer in failures:
            print(f"  {task_id}: {answer[:80]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
