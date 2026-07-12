"""Run the agent's REAL pipeline locally over the public validation tasks.

Accuracy-gate diagnosis (no Docker, no /input mounts, MOCK_FIREWORKS=1):
loads config, the local model, the classifier and the solver exactly as
main.py does, solves eval/validation_tasks.json, and writes
eval/validation_results.json with the full answer + category + source
per task. Content check only — math/logic route to the mocked
Fireworks client here, so their answers are meaningless placeholders
that merely confirm routing.

Usage (from agent/, with MOCK_FIREWORKS=1 and dummy FIREWORKS_* env):
    python eval/run_local_validation.py
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fireworks_client import make_client  # noqa: E402
from io_utils import load_config, load_tasks  # noqa: E402
from local_model import make_local_model  # noqa: E402
from solve import solve_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TASKS_PATH = Path(__file__).resolve().parent / "validation_tasks.json"
RESULTS_PATH = Path(__file__).resolve().parent / "validation_results.json"


def main() -> int:
    if os.environ.get("MOCK_FIREWORKS") != "1":
        sys.exit("Refusing to run without MOCK_FIREWORKS=1 (this is an "
                 "offline content diagnosis; no real calls).")

    config = {
        "allowed_models": [
            m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()
        ],
        "api_key": os.environ["FIREWORKS_API_KEY"],
        "base_url": os.environ["FIREWORKS_BASE_URL"],
    }
    config.update(load_config())

    tasks = load_tasks(str(TASKS_PATH))
    client = make_client(config)
    local_model = make_local_model(config)
    if local_model is None:
        sys.exit("No local model loaded — download the GGUF per models/README.md")

    results = solve_all(tasks, client, config, local_model=local_model)

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Local usage: {local_model.stats.summary()}")
    for r in results:
        print(f"\n--- {r['task_id']} [{r['category']} via {r['source']}]")
        print(r["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
