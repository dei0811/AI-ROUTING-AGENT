"""Routing behavior: local-first, escalation, and the budget flip.

Uses the offline mocks, so these tests assert where answers come from
(call logs) — the property the token ranking actually depends on.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fireworks_client import MockFireworksClient
from local_model import MockLocalModel
from solve import (
    FIREWORKS, LOCAL, clean_answer, resolve_model_tiers, resolve_route,
    solve_all, solve_task,
)

ALLOWED = ["mock-8b", "mock-70b"]
BASE_CONFIG = {
    "allowed_models": ALLOWED,
    "escalate_malformed_local": True,
    "local_code_exec_categories": ["math"],
}


def make_parts(local_responses=None, fw_responses=None):
    client = MockFireworksClient(ALLOWED, responses=fw_responses)
    local = MockLocalModel(responses=local_responses)
    tiers = resolve_model_tiers(ALLOWED)
    return client, local, tiers


class TestResolveRoute(unittest.TestCase):
    def test_defaults_follow_spec_table(self):
        local = MockLocalModel()
        self.assertEqual(resolve_route("factual", {}, local), LOCAL)
        self.assertEqual(resolve_route("code_gen", {}, local), FIREWORKS)
        self.assertEqual(resolve_route("code_debug", {}, local), FIREWORKS)

    def test_no_local_model_forces_fireworks(self):
        self.assertEqual(resolve_route("factual", {}, None), FIREWORKS)

    def test_config_override(self):
        local = MockLocalModel()
        config = {"category_routes": {"factual": "fireworks"}}
        self.assertEqual(resolve_route("factual", config, local), FIREWORKS)


class TestSolveTask(unittest.TestCase):
    def test_factual_answered_locally_zero_fireworks_calls(self):
        client, local, tiers = make_parts(local_responses=["Canberra"])
        result = solve_task(
            {"task_id": "t1", "prompt": "What is the capital of Australia?"},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["answer"], "Canberra")
        self.assertEqual(result["source"], "local")
        self.assertEqual(client.tokens.summary()["calls"], 0)

    def test_math_local_emits_code_and_executes(self):
        client, local, tiers = make_parts(
            local_responses=["```python\nprint(204 - 60)\n```"],
        )
        result = solve_task(
            {"task_id": "t1", "prompt": "Calculate 240 - 36 - 60"},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["answer"], "144")
        self.assertEqual(result["source"], "local")
        self.assertEqual(client.tokens.summary()["calls"], 0)

    def test_code_gen_goes_to_fireworks(self):
        client, local, tiers = make_parts(
            fw_responses=["```python\ndef f():\n    return 1\n```"],
        )
        result = solve_task(
            {"task_id": "t1", "prompt": "Write a Python function named f returning 1."},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["source"], "fireworks")
        self.assertEqual(len(local.call_log), 0)
        self.assertEqual(client.tokens.summary()["calls"], 1)

    def test_malformed_local_escalates_to_fireworks(self):
        # Sentiment must be one label; the local mock rambles, so one
        # cheap Fireworks call finishes the job.
        client, local, tiers = make_parts(
            local_responses=["It's hard to say, could be good or bad."],
            fw_responses=["positive"],
        )
        result = solve_task(
            {"task_id": "t1", "prompt": "Classify the sentiment of this review: 'Loved it.'"},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["answer"], "positive")
        self.assertEqual(result["source"], "local->fireworks")

    def test_malformed_local_without_escalation_ships_best_effort(self):
        client, local, tiers = make_parts(
            local_responses=["It's hard to say, could be good or bad."],
        )
        config = dict(BASE_CONFIG, escalate_malformed_local=False)
        result = solve_task(
            {"task_id": "t1", "prompt": "Classify the sentiment of this review: 'Loved it.'"},
            client, local, tiers, config,
        )
        self.assertEqual(result["source"], "local")
        self.assertEqual(client.tokens.summary()["calls"], 0)
        self.assertTrue(result["answer"])

    def test_empty_prompt_yields_empty_answer(self):
        client, local, tiers = make_parts()
        result = solve_task(
            {"task_id": "t1", "prompt": " "}, client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["source"], "none")


class TestSolveAll(unittest.TestCase):
    TASKS = [
        {"task_id": "t1", "prompt": "What is the capital of Australia?"},
        {"task_id": "t2", "prompt": "What is the capital of France?"},
    ]

    def test_exhausted_budget_flips_local_tasks_to_fireworks(self):
        client, local, _ = make_parts(fw_responses=["Canberra", "Paris"])
        config = dict(BASE_CONFIG, global_budget_s=0)
        results = solve_all(self.TASKS, client, config, local_model=local)
        self.assertEqual([r["source"] for r in results],
                         [FIREWORKS, FIREWORKS])
        self.assertEqual(len(local.call_log), 0)

    def test_no_local_model_all_fireworks(self):
        client = MockFireworksClient(ALLOWED, responses=["Canberra"])
        results = solve_all(self.TASKS, client, dict(BASE_CONFIG), local_model=None)
        self.assertEqual([r["source"] for r in results],
                         [FIREWORKS, FIREWORKS])

    def test_all_local_zero_fireworks_tokens(self):
        client, local, _ = make_parts(local_responses=["Canberra"])
        results = solve_all(self.TASKS, client, dict(BASE_CONFIG), local_model=local)
        self.assertEqual([r["source"] for r in results], [LOCAL, LOCAL])
        self.assertEqual(client.tokens.summary()["total_tokens"], 0)


class TestCleanAnswer(unittest.TestCase):
    def test_sentiment_label_extracted(self):
        ok, cleaned = clean_answer("sentiment", "The sentiment is Positive.")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "positive")

    def test_ner_json_normalized(self):
        ok, cleaned = clean_answer("ner", 'Sure! ```json\n{"people": ["Ada"]}\n```')
        self.assertTrue(ok)
        self.assertEqual(cleaned, '{"people":["Ada"]}')

    def test_code_fence_stripped(self):
        ok, cleaned = clean_answer("code_gen", "```python\ndef f(): pass\n```")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "def f(): pass")


if __name__ == "__main__":
    unittest.main()
