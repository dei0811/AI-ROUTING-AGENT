"""Task loading and result writing for the I/O contract.

Contract (see CLAUDE_AGENT_SPEC.md §3):
- Input:  /input/tasks.json   -> list of {"task_id": str, "prompt": str}
- Output: /output/results.json -> list of {"task_id": str, "answer": str}

Every task_id present in the input must appear exactly once in the
output, even when solving fails (best-effort or empty answer). The
output must always be valid JSON or the submission scores zero.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path: str = None) -> dict:
    """Load tuning config (concurrency, per-category caps/tiers).

    Missing file is not an error — code-level defaults apply — because
    the scored run must never die over an optional tuning file.

    Args:
        path: Config file path; defaults to config.json next to the code.

    Returns:
        Dict of config values ({} if the file is absent or invalid).
    """
    file_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not file_path.exists():
        logger.warning("No config file at %s; using code defaults", file_path)
        return {}

    try:
        with open(file_path, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError(f"config must be a JSON object, got {type(config).__name__}")
    except (ValueError, OSError):
        logger.exception("Invalid config at %s; using code defaults", file_path)
        return {}

    logger.info("Loaded config from %s", file_path)
    return config


def load_tasks(path: str) -> list:
    """Load and validate the task batch from ``path``.

    Malformed entries are handled best-effort instead of aborting the
    batch: an entry with a usable ``task_id`` but a bad/missing prompt
    is kept (with an empty prompt) so its id can still be covered in
    the output; an entry with no usable ``task_id`` is dropped since
    there is no id to answer under.

    Args:
        path: Path to the tasks JSON file.

    Returns:
        List of dicts, each with at least ``task_id`` (non-empty str)
        and ``prompt`` (str, possibly empty). Extra keys are preserved.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is not a list.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input tasks file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input file is not valid JSON: {file_path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Input file must contain a JSON array of tasks, got {type(data).__name__}"
        )

    tasks = []
    seen_ids = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Skipping task at index %d: not an object (%r)", index, item)
            continue

        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("Skipping task at index %d: missing/invalid task_id", index)
            continue

        if task_id in seen_ids:
            logger.warning("Duplicate task_id %r at index %d; keeping first occurrence", task_id, index)
            continue
        seen_ids.add(task_id)

        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            logger.warning("Task %r has missing/invalid prompt; using empty prompt", task_id)
            prompt = ""

        task = dict(item)
        task["task_id"] = task_id
        task["prompt"] = prompt
        tasks.append(task)

    logger.info("Loaded %d valid tasks (of %d entries) from %s", len(tasks), len(data), file_path)
    return tasks


def write_results(path: str, results: list, expected_ids: list) -> None:
    """Validate and write the results file.

    Guarantees the output covers every expected task_id exactly once:
    missing ids are filled with an empty answer, duplicates and unknown
    ids are dropped, and non-string answers are coerced to strings.

    Args:
        path: Destination path for results.json.
        results: List of {"task_id": ..., "answer": ...} dicts.
        expected_ids: Every task_id that must appear in the output.

    Raises:
        OSError: If the file cannot be written.
    """
    by_id = {}
    for item in results:
        if not isinstance(item, dict):
            logger.warning("Dropping malformed result entry: %r", item)
            continue
        task_id = item.get("task_id")
        if task_id not in expected_ids:
            logger.warning("Dropping result with unknown task_id: %r", task_id)
            continue
        if task_id in by_id:
            logger.warning("Duplicate result for task_id %r; keeping first", task_id)
            continue

        answer = item.get("answer")
        if answer is None:
            answer = ""
        elif not isinstance(answer, str):
            answer = str(answer)
        by_id[task_id] = answer

    final = []
    for task_id in expected_ids:
        if task_id not in by_id:
            logger.warning("No result produced for task_id %r; writing empty answer", task_id)
        final.append({"task_id": task_id, "answer": by_id.get(task_id, "")})

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)

    logger.info("Wrote %d results to %s", len(final), file_path)
