"""Entrypoint for the token-efficient general-purpose agent.

Reads /input/tasks.json, solves every task through Fireworks AI, and
writes /output/results.json. Exit code 0 on success, non-zero on any
fatal error. Startup must stay light (<60 s budget): no heavy imports
or setup before reading the input.

Phase 2: single general prompt per task through the (mockable) client.
"""

import logging
import os
import sys
import time

_START = time.monotonic()  # startup counts against the global budget

from fireworks_client import make_client
from io_utils import load_config, load_tasks, write_results
from solve import solve_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "/input/tasks.json"
DEFAULT_OUTPUT = "/output/results.json"

REQUIRED_ENV_VARS = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")


def read_env_config() -> dict:
    """Read the harness-injected environment variables.

    Returns:
        Dict with ``api_key``, ``base_url`` and ``allowed_models``
        (non-empty list of model ids parsed from ALLOWED_MODELS).

    Raises:
        RuntimeError: If any required variable is missing/empty, or if
            ALLOWED_MODELS contains no model ids.
    """
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". They are injected by the judging harness; for local runs "
            "set them manually (dummy values are fine before Phase 1)."
        )

    allowed_models = [
        model.strip()
        for model in os.environ["ALLOWED_MODELS"].split(",")
        if model.strip()
    ]
    if not allowed_models:
        raise RuntimeError("ALLOWED_MODELS is set but contains no model ids.")

    return {
        "api_key": os.environ["FIREWORKS_API_KEY"],
        "base_url": os.environ["FIREWORKS_BASE_URL"],
        "allowed_models": allowed_models,
    }


def main() -> int:
    input_path = os.environ.get("INPUT_FILE", DEFAULT_INPUT)
    output_path = os.environ.get("OUTPUT_FILE", DEFAULT_OUTPUT)

    try:
        config = read_env_config()
        config.update(load_config(os.environ.get("CONFIG_FILE")))
        logger.info("Allowed models: %s", config["allowed_models"])

        tasks = load_tasks(input_path)
        client = make_client(config)
        results = solve_all(tasks, client, config, start_time=_START)
        write_results(output_path, results, expected_ids=[t["task_id"] for t in tasks])
        logger.info("Token usage: %s", client.tokens.summary())
    except Exception:
        logger.exception("Fatal error; no valid results written")
        return 1

    logger.info("Done: %d tasks answered.", len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
