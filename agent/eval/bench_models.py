"""Benchmark candidate local GGUF models under grading-box limits.

Per BENCHMARK_LOCAL_MODELS.md: for each candidate in models/bench/,
measure — in a subprocess, so peak RSS is per-model and a crash cannot
kill the run — load time, peak RSS, prefill/decode speed at n_threads=2
(2-vCPU emulation), and judge pass-rate on the dev set, then apply the
hard gates and pick a winner.

Quality runs the six locally-served categories (factual, math,
sentiment, summarization, ner, logic) through the real solve pipeline
(route forced LOCAL, escalation off) with the offline heuristic judge,
so the bench needs no Fireworks key. Code categories are routed to
Fireworks in production and are not part of local model selection.

Usage (from agent/):
    python eval/bench_models.py               # bench all + report + cleanup
    AUTO_DELETE=0 python eval/bench_models.py # stop after printing the plan
    KEEP_FALLBACK=1 ...                       # also keep the smallest RAM-fitting model
    python eval/bench_models.py --measure path/to/model.gguf [--no-think]  # child mode

Outputs (written BEFORE any deletion):
    eval/benchmark_results.json
    eval/BENCHMARK_REPORT.md
"""

import argparse
import ctypes
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")  # before llama_cpp loads

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "eval"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("bench")

BENCH_DIR = AGENT_DIR / "models" / "bench"
SHIPPED_DIR = AGENT_DIR / "models"
RESULTS_JSON = AGENT_DIR / "eval" / "benchmark_results.json"
REPORT_MD = AGENT_DIR / "eval" / "BENCHMARK_REPORT.md"

# 2-vCPU grading-box emulation (spec §3.1).
BENCH_THREADS = 2
BENCH_CTX = 2048
# Hard per-generation ceiling inside a task; generation streams and
# truncates at the deadline, so a slow model cannot hang the bench.
BENCH_TASK_BUDGET_S = 30.0
CHILD_TIMEOUT_S = 2700  # whole-candidate ceiling (load + probe + 30 tasks)

# Hard gates (spec §1).
MAX_PEAK_RSS_GB = 3.4      # 4 GB minus ~0.5 GB agent headroom
MAX_LOAD_S = 60.0
# "Category-appropriate answer < 30 s": floor of usable answer tokens
# within the per-request limit (sentiment label, short fact, compact
# NER JSON, one-sentence summary all fit in 64).
MIN_TOKENS_30S = 64

LOCAL_CATEGORIES = ("factual", "math", "sentiment", "summarization", "ner", "logic")
OBJECTIVE_CATEGORIES = ("math", "ner", "sentiment")

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

# name, bench subdir, needs /no_think (thinking-capable models).
CANDIDATES = (
    ("Qwen3.5-4B", "qwen3.5-4b", True),
    ("SmolLM3-3B", "smollm3-3b", True),
    ("Gemma-3-4B-it-QAT", "gemma-3-4b-it-qat", False),
    ("Phi-4-mini-instruct", "phi-4-mini", False),
    ("Llama-3.2-3B-Instruct", "llama-3.2-3b", False),
)
BASELINE_NAME = "Qwen2.5-0.5B"
BASELINE_GLOB = str(SHIPPED_DIR / "qwen2.5-0.5b*.gguf")

# Candidates that could not be resolved at all (spec §2: log, don't fail).
PRE_SKIPPED = (
    {
        "model": "Qwen3.5-2B-Instruct",
        "verdict": "SKIPPED",
        "reason": "No such model on Hugging Face (Qwen3.5 family starts at 4B); "
                  "benchmarked Qwen3.5-4B as the nearest Qwen candidate instead.",
    },
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


# ---------------------------------------------------------------- child mode

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_bytes() -> int:
    """Peak RSS of this process (Windows PeakWorkingSet / POSIX ru_maxrss)."""
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p,
                             ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                             ctypes.c_uint32]
        get_info.restype = ctypes.c_int
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p

        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(pmc.PeakWorkingSetSize)
    import resource
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024)


class NoThinkLocalModel:
    """LocalModel wrapper for thinking-capable candidates: prepends the
    /no_think soft switch and strips any <think> block that slips
    through (an unclosed block truncates to nothing — correctly scoring
    a model that spent its whole token cap thinking)."""

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


def measure_one(model_path: str, no_think: bool) -> dict:
    """Child-process measurement of a single GGUF (spec §3 steps 1-5)."""
    from io_utils import load_config
    from judge import heuristic_judge
    from local_model import LocalModel
    from run_eval import load_dev_set
    from solve import LOCAL, resolve_model_tiers, solve_task

    record = {"model_path": model_path,
              "file_size_gb": round(os.path.getsize(model_path) / 1e9, 2)}

    # -- load (timed)
    started = time.monotonic()
    lm = LocalModel(model_path, n_ctx=BENCH_CTX, n_threads=BENCH_THREADS,
                    max_tokens_cap=256)
    record["load_s"] = round(time.monotonic() - started, 1)
    model = NoThinkLocalModel(lm) if no_think else lm

    # -- warmup (first call pays one-time init costs)
    model.generate([{"role": "user", "content": "Say OK."}], max_tokens=8,
                   deadline=time.monotonic() + 60)

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
        if time.monotonic() - t0 > 120:  # probe safety net
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

    record["peak_rss_gb"] = round(peak_rss_bytes() / 1e9, 2)
    record["local_stats"] = lm.stats.summary()
    return record


# --------------------------------------------------------------- parent mode

def run_child(name: str, gguf: str, no_think: bool) -> dict:
    """Run one candidate's measurement in a subprocess; DISCARD on error."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--measure", gguf]
    if no_think:
        cmd.append("--no-think")
    env = dict(os.environ, OMP_NUM_THREADS=str(BENCH_THREADS))
    print(f"\n=== {name}: measuring {Path(gguf).name} ...", flush=True)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=CHILD_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"model": name, "verdict": "DISCARD",
                "reason": f"benchmark exceeded {CHILD_TIMEOUT_S}s"}

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"model": name, "verdict": "DISCARD",
                "reason": "measurement crashed: " + " | ".join(tail)}
    try:
        record = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"model": name, "verdict": "DISCARD",
                "reason": "measurement produced no JSON"}
    record["model"] = name
    return record


def apply_gates(record: dict, baseline: dict) -> None:
    """Set fits_4gb / verdict / reason on a measured record (spec §1)."""
    if record.get("verdict") == "DISCARD":  # crashed/timed out earlier
        record.setdefault("fits_4gb", False)
        return

    record["fits_4gb"] = record["peak_rss_gb"] < MAX_PEAK_RSS_GB
    reasons = []
    if not record["fits_4gb"]:
        reasons.append(f"peak RSS {record['peak_rss_gb']} GB > {MAX_PEAK_RSS_GB} GB")
    if record["load_s"] >= MAX_LOAD_S:
        reasons.append(f"load {record['load_s']}s >= {MAX_LOAD_S}s")
    if record["est_tokens_30s"] < MIN_TOKENS_30S:
        reasons.append(
            f"only ~{record['est_tokens_30s']} tokens fit in 30s (< {MIN_TOKENS_30S})"
        )

    if baseline is not None and record is not baseline:
        if record["overall_pass"] <= baseline["overall_pass"]:
            reasons.append(
                f"overall {record['overall_pass']:.2f} not above baseline "
                f"{baseline['overall_pass']:.2f}"
            )
        regressed = [
            cat for cat in OBJECTIVE_CATEGORIES
            if record["pass_by_category"].get(cat, 0)
            < baseline["pass_by_category"].get(cat, 0)
        ]
        if regressed:
            reasons.append("regresses " + "/".join(regressed) + " vs baseline")

    if record is baseline:
        record["verdict"] = "BASELINE" if not reasons else "DISCARD"
    else:
        record["verdict"] = "KEEP" if not reasons else "DISCARD"
    record["reason"] = "; ".join(reasons)


def pick_winner(records: list) -> tuple:
    """(winner, forced) per spec §4; forced=True when no candidate KEEPs."""
    keepers = [r for r in records if r.get("verdict") == "KEEP"]
    if keepers:
        keepers.sort(key=lambda r: (-r["overall_pass"], -r["decode_tok_s"],
                                    r["file_size_gb"]))
        return keepers[0], False

    fitting = [r for r in records
               if r.get("fits_4gb") and "overall_pass" in r]
    if not fitting:
        return None, True
    fitting.sort(key=lambda r: -r["overall_pass"])
    return fitting[0], True


def write_report(records: list, winner: dict, forced: bool, gemma_note: str) -> None:
    RESULTS_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    def cell(r, key, fmt="{}"):
        return fmt.format(r[key]) if key in r else "—"

    lines = [
        "# Local model benchmark — Track 1",
        "",
        f"Bench box: Windows host, llama.cpp `n_threads={BENCH_THREADS}` + "
        f"`OMP_NUM_THREADS={BENCH_THREADS}` (2-vCPU emulation), ctx {BENCH_CTX}, "
        "temperature 0, thinking disabled. Peak RSS is the per-model subprocess "
        "peak working set — a proxy for the 4 GB cgroup limit of the grading box. "
        "Quality = offline heuristic judge over the 30 locally-served dev tasks "
        "(factual, math, sentiment, summarization, ner, logic); code categories "
        "route to Fireworks in production and do not weigh on local selection.",
        "",
        "| model | size(GB) | load(s) | peakRAM(GB) | fits4GB | decode tok/s | "
        "est tok≤30s | overall pass | math | ner | sentiment | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(
        records,
        key=lambda r: (r.get("verdict") == "SKIPPED", -r.get("overall_pass", -1)),
    )
    for r in ranked:
        cats = r.get("pass_by_category", {})
        lines.append(
            f"| {r['model']} | {cell(r, 'file_size_gb')} | {cell(r, 'load_s')} | "
            f"{cell(r, 'peak_rss_gb')} | {cell(r, 'fits_4gb')} | "
            f"{cell(r, 'decode_tok_s')} | {cell(r, 'est_tokens_30s')} | "
            f"{cell(r, 'overall_pass')} | {cats.get('math', '—')} | "
            f"{cats.get('ner', '—')} | {cats.get('sentiment', '—')} | "
            f"{r.get('verdict', '?')}{': ' + r['reason'] if r.get('reason') else ''} |"
        )

    lines.append("")
    if winner is None:
        lines.append("**No candidate fit the grading box — investigate before shipping.**")
    else:
        lines.append(
            f"**Winner: {winner['model']}** — overall pass "
            f"{winner['overall_pass']:.2f}, {winner['decode_tok_s']} decode tok/s, "
            f"{winner['file_size_gb']} GB, peak RSS {winner['peak_rss_gb']} GB, "
            f"load {winner['load_s']}s."
        )
        if forced:
            lines.append(
                "**WARNING: no candidate passed every hard gate.** This is the "
                "best RAM-fitting model; the 30 s limit forces shorter outputs — "
                "lower `local_max_tokens_cap` accordingly."
            )
    if gemma_note:
        lines.append("")
        lines.append(gemma_note)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {RESULTS_JSON} and {REPORT_MD}")


def cleanup(records: list, winner: dict) -> None:
    """Spec §6: plan, copy winner to shipped path, delete losers."""
    keep_fallback = os.environ.get("KEEP_FALLBACK") == "1"
    fallback = None
    if keep_fallback:
        fitting = [r for r in records
                   if r.get("fits_4gb") and r is not winner and "file_size_gb" in r]
        if fitting:
            fallback = min(fitting, key=lambda r: r["file_size_gb"])

    kept_paths = {Path(winner["model_path"]).resolve()}
    if fallback:
        kept_paths.add(Path(fallback["model_path"]).resolve())

    delete_paths = []
    for pattern in (str(BENCH_DIR / "*" / "*.gguf"), str(SHIPPED_DIR / "*.gguf")):
        for f in glob.glob(pattern):
            p = Path(f).resolve()
            if p not in kept_paths:
                delete_paths.append(p)

    freed = sum(p.stat().st_size for p in delete_paths)
    print("\n=== Cleanup plan ===")
    print(f"KEEP  (winner):   {winner['model_path']}")
    if fallback:
        print(f"KEEP  (fallback): {fallback['model_path']}")
    for p in delete_paths:
        print(f"DELETE ({p.stat().st_size / 1e9:.2f} GB): {p}")
    print(f"Would free ~{freed / 1e9:.2f} GB")

    if os.environ.get("AUTO_DELETE", "1") == "0":
        print("AUTO_DELETE=0 -> stopping after the plan; nothing deleted.")
        return

    # Copy kept weights into the shipped dir BEFORE deleting anything.
    shipped = []
    for record in ([winner] + ([fallback] if fallback else [])):
        src = Path(record["model_path"]).resolve()
        dst = (SHIPPED_DIR / src.name).resolve()
        if src != dst:
            print(f"Copying {src.name} -> {dst}")
            shutil.copy2(src, dst)
        shipped.append(dst)

    config_path = AGENT_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["local_model_path"] = f"models/{shipped[0].name}"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"config.json local_model_path -> models/{shipped[0].name}")

    for p in delete_paths:
        p.unlink()
    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)  # bench copies of kept models included
    print(f"Freed ~{freed / 1e9:.2f} GB")
    print("Final shipped model(s): " + ", ".join(str(s) for s in shipped))


def main() -> int:
    # Windows consoles default to cp1252, which cannot print "≤" etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", metavar="GGUF",
                        help="child mode: measure one model, print JSON")
    parser.add_argument("--no-think", action="store_true",
                        help="child mode: apply the /no_think switch")
    args = parser.parse_args()

    if args.measure:
        record = measure_one(args.measure, args.no_think)
        print(json.dumps(record))
        return 0

    records = []

    baseline_files = glob.glob(BASELINE_GLOB)
    baseline = None
    if baseline_files:
        baseline = run_child(BASELINE_NAME, baseline_files[0], no_think=False)
    else:
        records.append({"model": BASELINE_NAME, "verdict": "SKIPPED",
                        "reason": "baseline weights not found in models/"})

    for name, subdir, no_think in CANDIDATES:
        ggufs = glob.glob(str(BENCH_DIR / subdir / "*.gguf"))
        if not ggufs:
            records.append({"model": name, "verdict": "SKIPPED",
                            "reason": f"no GGUF in models/bench/{subdir} "
                                      "(download failed or repo unavailable)"})
            continue
        records.append(run_child(name, ggufs[0], no_think))

    if baseline is not None:
        records.append(baseline)
        apply_gates(baseline, baseline)
    for record in records:
        if record.get("verdict") == "SKIPPED" or record is baseline:
            continue
        apply_gates(record, baseline)

    records.extend(PRE_SKIPPED)

    winner, forced = pick_winner(records)

    gemma_note = ""
    gemma = next((r for r in records if r["model"].startswith("Gemma-3-4B")), None)
    if (winner and gemma and gemma.get("verdict") == "KEEP"
            and gemma is not winner
            and winner["overall_pass"] - gemma["overall_pass"] <= 0.02):
        gemma_note = (
            "**Gemma-challenge alternative:** Gemma-3-4B-it-QAT is within 2 points "
            "of the winner and passes all gates; pairing it locally with Gemma-4 "
            "on Fireworks strengthens a Best-Use-of-Gemma entry."
        )

    write_report(records, winner, forced, gemma_note)

    if winner is None:
        print("\nNO WINNER — nothing deleted.")
        return 1

    print(f"\nWINNER: {winner['model']} (pass={winner['overall_pass']:.2f}, "
          f"decode={winner['decode_tok_s']} tok/s, size={winner['file_size_gb']} GB)")

    cleanup(records, winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
