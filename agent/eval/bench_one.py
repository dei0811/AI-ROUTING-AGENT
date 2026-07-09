"""Measure ONE local GGUF — runs INSIDE the capped bench container.

The orchestrator (eval/bench_models.py) launches this under
``--memory=4g --memory-swap=4g --cpus=2 --platform linux/amd64`` with
the candidate mounted read-only at /model. Surviving the full run in
here IS the RAM verdict: if the model + KV + agent code exceed 4 GB the
kernel OOM-kills us (exit 137) and the orchestrator records the fact.

Env knobs (set by the orchestrator):
    CTX  — n_ctx for the load (default 1536)
    KV   — KV-cache type: q8_0 (default) / q4_0 / f16
    OMP_NUM_THREADS — always 2 for the bench

Writes /out/<name>.json and exits 0 on success.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/eval")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # host dev runs
sys.path.insert(0, str(Path(__file__).resolve().parent))

BENCH_THREADS = 2
# Hard per-generation ceiling; generation streams and truncates at the
# deadline, so a slow model cannot hang the bench.
BENCH_TASK_BUDGET_S = 30.0

LOCAL_CATEGORIES = ("factual", "math", "sentiment", "summarization", "ner", "logic")

SPEED_PROBE_PROMPT = (
    "The Amazon rainforest spans nine countries and holds roughly ten percent "
    "of the planet's known species. Its trees cycle enormous volumes of water "
    "into the atmosphere, seeding rainfall far beyond the basin itself, while "
    "its soils and biomass store carbon accumulated over centuries. Clearing "
    "for cattle ranching, soy farming and mining has pushed parts of the forest "
    "toward a drier, savanna-like state, and researchers warn that continued "
    "loss could tip the system past recovery. Describe the main ecological "
    "functions of the Amazon rainforest and the pressures it faces."
)
SPEED_PROBE_TOKENS = 128

_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


class NoThinkLocalModel:
    """Wrapper for thinking-capable candidates: prepends the /no_think
    soft switch and strips any <think> block that slips through (an
    unclosed block truncates to nothing — correctly scoring a model
    that spent its whole token cap thinking)."""

    def __init__(self, inner):
        self._inner = inner
        self.stats = inner.stats
        self.max_tokens_cap = inner.max_tokens_cap

    def generate(self, messages, max_tokens, deadline=None):
        messages = [dict(m) for m in messages]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = "/no_think\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": "/no_think"})
        raw = self._inner.generate(messages, max_tokens, deadline)
        return _THINK_RE.sub("", raw).strip()


def cgroup_peak_gb() -> float:
    """Peak memory charged to this container's cgroup (informational —
    the authoritative RAM verdict is OOMKilled true/false)."""
    for path in ("/sys/fs/cgroup/memory.peak",                      # v2
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):  # v1
        try:
            return round(int(Path(path).read_text().strip()) / 1e9, 2)
        except (OSError, ValueError):
            continue
    return -1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the GGUF")
    parser.add_argument("--name", required=True, help="candidate name")
    parser.add_argument("--no-think", action="store_true",
                        help="apply the /no_think switch")
    args = parser.parse_args()

    ctx = int(os.environ.get("CTX", "1536"))
    kv = os.environ.get("KV", "q8_0")

    from io_utils import load_config
    from judge import heuristic_judge
    from local_model import LocalModel
    from run_eval import load_dev_set
    from solve import LOCAL, resolve_model_tiers, solve_task

    record = {
        "model": args.name,
        "file_size_gb": round(os.path.getsize(args.model) / 1e9, 2),
        "ctx": ctx,
        "kv": kv,
    }

    # -- load (timed)
    started = time.monotonic()
    lm = LocalModel(args.model, n_ctx=ctx, n_threads=BENCH_THREADS,
                    max_tokens_cap=256, kv_type=kv)
    record["load_s"] = round(time.monotonic() - started, 1)
    model = NoThinkLocalModel(lm) if args.no_think else lm

    # -- warmup (first call pays one-time init costs)
    model.generate([{"role": "user", "content": "Say OK."}], max_tokens=8,
                   deadline=time.monotonic() + 120)

    # -- speed probe: stream 128 tokens; first token time ~= prefill
    prompt_tokens = len(lm._llama.tokenize(SPEED_PROBE_PROMPT.encode("utf-8")))
    t0 = time.monotonic()
    t_first = None
    n_tokens = 0
    stream = lm._llama.create_chat_completion(
        messages=[{"role": "user", "content": SPEED_PROBE_PROMPT}],
        max_tokens=SPEED_PROBE_TOKENS, temperature=0.0, stream=True,
    )
    for chunk in stream:
        if chunk["choices"][0]["delta"].get("content"):
            if t_first is None:
                t_first = time.monotonic()
            n_tokens += 1
        if time.monotonic() - t0 > 180:  # probe safety net
            break
    t_end = time.monotonic()

    prefill_s = (t_first or t_end) - t0
    decode_s = max(t_end - (t_first or t_end), 1e-6)
    record["prefill_tok_s"] = round(prompt_tokens / max(prefill_s, 1e-6), 1)
    record["decode_tok_s"] = round(max(n_tokens - 1, 0) / decode_s, 1)
    record["est_tokens_30s"] = max(
        0, int(record["decode_tok_s"] * (30 - prefill_s - 1))
    )

    # -- quality: real solve pipeline, route forced LOCAL, offline judge
    config = load_config()
    config.update({
        "allowed_models": ["offline-bench"],   # never called: escalation off
        "escalate_malformed_local": False,
        "local_task_budget_s": BENCH_TASK_BUDGET_S,
    })
    tiers = resolve_model_tiers(config["allowed_models"])
    tasks = [t for t in load_dev_set() if t["category"] in LOCAL_CATEGORIES]

    per_category = {}
    for task in tasks:
        result = solve_task(task, None, model, tiers, config, route=LOCAL)
        passed = heuristic_judge(task["category"], result["answer"], task["expected"])
        n, p = per_category.get(task["category"], (0, 0))
        per_category[task["category"]] = (n + 1, p + (1 if passed else 0))

    record["pass_by_category"] = {
        cat: round(p / n, 2) for cat, (n, p) in sorted(per_category.items())
    }
    total_n = sum(n for n, _ in per_category.values())
    total_p = sum(p for _, p in per_category.values())
    record["overall_pass"] = round(total_p / total_n, 2) if total_n else 0.0

    record["cgroup_peak_gb"] = cgroup_peak_gb()
    record["local_stats"] = lm.stats.summary()
    record["completed_batch"] = True

    out_dir = Path(os.environ.get("OUT_DIR", "/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8",
    )
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
