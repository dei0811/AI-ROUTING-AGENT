"""I/O contract: every input task_id covered exactly once, valid JSON."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import load_tasks, write_results


class TestLoadTasks(unittest.TestCase):
    def _write(self, data) -> str:
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, f)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_valid_batch(self):
        path = self._write([{"task_id": "t1", "prompt": "hi"}])
        tasks = load_tasks(path)
        self.assertEqual(tasks, [{"task_id": "t1", "prompt": "hi"}])

    def test_bad_prompt_kept_with_empty_string(self):
        path = self._write([{"task_id": "t1", "prompt": 5}])
        self.assertEqual(load_tasks(path)[0]["prompt"], "")

    def test_missing_id_dropped_duplicates_deduped(self):
        path = self._write([
            {"prompt": "no id"},
            {"task_id": "t1", "prompt": "a"},
            {"task_id": "t1", "prompt": "b"},
        ])
        tasks = load_tasks(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["prompt"], "a")

    def test_non_list_raises(self):
        path = self._write({"task_id": "t1"})
        with self.assertRaises(ValueError):
            load_tasks(path)


class TestWriteResults(unittest.TestCase):
    def test_covers_every_expected_id(self):
        out = Path(tempfile.mkdtemp()) / "results.json"
        write_results(
            str(out),
            [
                {"task_id": "t2", "answer": "b", "category": "x", "source": "local"},
                {"task_id": "ghost", "answer": "?"},
            ],
            expected_ids=["t1", "t2"],
        )
        written = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(written, [
            {"task_id": "t1", "answer": ""},
            {"task_id": "t2", "answer": "b"},
        ])


if __name__ == "__main__":
    unittest.main()
