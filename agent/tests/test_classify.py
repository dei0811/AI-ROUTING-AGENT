"""Category heuristics: one representative prompt per category, plus
the ordering rules that keep overlapping cues from misrouting."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import classify as c


class TestClassify(unittest.TestCase):
    def test_one_prompt_per_category(self):
        cases = {
            c.FACTUAL: "What is the capital of Australia?",
            c.MATH: "A store has 240 items. 15% sell on Monday, then 60 more on Tuesday. How many remain?",
            c.SENTIMENT: "Classify the sentiment of this review: 'Great value, would buy again.'",
            c.SUMMARIZATION: "Summarize in one sentence: The committee met on Thursday...",
            c.NER: "Extract the person, organization and location names from: 'Maria Sanchez of Fireworks AI spoke in Berlin.'",
            c.CODE_DEBUG: "Fix the bug:\n```python\ndef get_max(xs):\n    return min(xs)\n```",
            c.LOGIC: "Sam, Jo and Lee each have one pet. If Sam has the dog then Jo has the cat. Who has the fish?",
            c.CODE_GEN: "Write a Python function that returns the second-largest value, handling duplicates.",
        }
        for expected, prompt in cases.items():
            self.assertEqual(c.classify(prompt), expected, msg=prompt)

    def test_code_plus_error_cues_is_debug_not_gen(self):
        prompt = "This throws a TypeError, please fix:\n```python\nlen(3)\n```"
        self.assertEqual(c.classify(prompt), c.CODE_DEBUG)

    def test_write_a_function_to_sum_is_code_gen_not_math(self):
        prompt = "Write a function to sum a list of integers in Python."
        self.assertEqual(c.classify(prompt), c.CODE_GEN)

    def test_empty_prompt_is_unknown(self):
        self.assertEqual(c.classify(""), c.UNKNOWN)
        self.assertEqual(c.classify("   "), c.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
