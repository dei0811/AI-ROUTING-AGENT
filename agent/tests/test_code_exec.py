"""Sandboxed runner for model-emitted code: success, failure, timeout."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_exec import extract_code, run_python


class TestExtractCode(unittest.TestCase):
    def test_fenced_block(self):
        text = "Here you go:\n```python\nprint(42)\n```\nEnjoy."
        self.assertEqual(extract_code(text), "print(42)")

    def test_bare_text_is_code(self):
        self.assertEqual(extract_code("print(1)"), "print(1)")

    def test_empty(self):
        self.assertEqual(extract_code(""), "")


class TestRunPython(unittest.TestCase):
    def test_success(self):
        ok, out = run_python("print(6 * 7)")
        self.assertTrue(ok)
        self.assertEqual(out, "42")

    def test_error_reports_stderr(self):
        ok, out = run_python("raise ValueError('boom')")
        self.assertFalse(ok)
        self.assertIn("boom", out)

    def test_no_output_is_failure(self):
        ok, out = run_python("x = 1")
        self.assertFalse(ok)
        self.assertEqual(out, "no output")

    def test_timeout(self):
        ok, out = run_python("while True: pass", timeout_s=1.0)
        self.assertFalse(ok)
        self.assertEqual(out, "timeout")


if __name__ == "__main__":
    unittest.main()
