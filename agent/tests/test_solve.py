"""Routing behavior: local-first, escalation, and the budget flip.

Uses the offline mocks, so these tests assert where answers come from
(call logs) — the property the token ranking actually depends on.
"""

import json
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

SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

ALLOWED = ["mock-8b", "mock-70b"]
BASE_CONFIG = {
    "allowed_models": ALLOWED,
    "escalate_malformed_local": True,
    "category_routes": {"math": "fireworks", "logic": "fireworks"},
    "code_exec_categories": [],
    "local_code_exec_categories": [],
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

    def test_math_goes_to_fireworks_direct(self):
        # v2 reroute: math answers come from Fireworks directly, no
        # emit-code detour and no local decode.
        client, local, tiers = make_parts(fw_responses=["144"])
        result = solve_task(
            {"task_id": "t1", "prompt": "Calculate 240 - 36 - 60"},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["answer"], "144")
        self.assertEqual(result["source"], "fireworks")
        self.assertEqual(len(local.call_log), 0)
        self.assertEqual(client.tokens.summary()["calls"], 1)

    def test_logic_goes_to_fireworks_direct(self):
        client, local, tiers = make_parts(fw_responses=["Dog"])
        result = solve_task(
            {"task_id": "t1", "prompt":
                "If Ana is taller than Bo and Bo is taller than Cy, "
                "deduce who is tallest."},
            client, local, tiers, BASE_CONFIG,
        )
        self.assertEqual(result["source"], "fireworks")
        self.assertEqual(len(local.call_log), 0)
        self.assertEqual(client.tokens.summary()["calls"], 1)

    def test_math_local_emit_code_mechanism_still_works(self):
        # The local emit-code path is unused by the shipped config but
        # remains available via config override; keep it covered.
        client, local, tiers = make_parts(
            local_responses=["```python\nprint(204 - 60)\n```"],
        )
        config = dict(
            BASE_CONFIG,
            category_routes={"math": "local"},
            local_code_exec_categories=["math"],
        )
        result = solve_task(
            {"task_id": "t1", "prompt": "Calculate 240 - 36 - 60"},
            client, local, tiers, config,
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


class TestShippedConfigRouting(unittest.TestCase):
    """Pin the routing table that ships in config.json (v2 reroute)."""

    @classmethod
    def setUpClass(cls):
        with open(SHIPPED_CONFIG_PATH, encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_math_and_logic_route_fireworks(self):
        local = MockLocalModel()
        for category in ("math", "logic", "code_debug", "code_gen"):
            self.assertEqual(
                resolve_route(category, self.config, local), FIREWORKS, category,
            )

    def test_four_local_categories_stay_local(self):
        local = MockLocalModel()
        for category in ("factual", "sentiment", "summarization", "ner"):
            self.assertEqual(
                resolve_route(category, self.config, local), LOCAL, category,
            )

    def test_code_exec_paths_disabled(self):
        # Explicit empty lists: absent keys would fall back to the code
        # defaults, which re-enable the emit-code path for math.
        self.assertEqual(self.config["code_exec_categories"], [])
        self.assertEqual(self.config["local_code_exec_categories"], [])

    def test_math_and_logic_use_a_general_tier_not_the_code_tier(self):
        tiers = self.config["category_tiers"]
        self.assertEqual(tiers["math"], "mid")
        self.assertEqual(tiers["logic"], "mid")


class TestCleanAnswer(unittest.TestCase):
    def test_sentiment_keeps_label_and_reason(self):
        ok, cleaned = clean_answer(
            "sentiment", "Positive. The reviewer praises the value.")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "Positive. The reviewer praises the value.")

    def test_sentiment_dual_sided_reason_is_well_formed(self):
        # Graded tasks demand reasons covering both sides; mentioning
        # "positive" and "negative" together must not read as malformed.
        text = ("Neutral: the review notes negative shipping issues but "
                "positive product quality.")
        ok, cleaned = clean_answer("sentiment", text)
        self.assertTrue(ok)
        self.assertEqual(cleaned, text)

    def test_sentiment_without_any_label_is_malformed(self):
        ok, _ = clean_answer("sentiment", "It's hard to say either way.")
        self.assertFalse(ok)

    def test_ner_truncated_array_recovers_all_complete_objects(self):
        # A token-capped completion cut this array mid-entity; every
        # complete inner object must survive cleaning.
        truncated = ('[\n{"text": "Google", "type": "ORGANIZATION"},\n'
                     '{"text": "Zurich", "type": "LOCATION"},\n{"text": "ETH')
        ok, cleaned = clean_answer("ner", truncated)
        self.assertTrue(ok)
        parsed = json.loads(cleaned)
        self.assertEqual(len(parsed), 2)
        self.assertIn({"text": "Zurich", "type": "LOCATION"}, parsed)

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
