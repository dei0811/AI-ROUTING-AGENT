"""Local model benchmark v2 — container-based (host orchestrator).

v1 measured peak RSS on the Windows host, which over-counts mmap'd
weights and does not map to the grading box's 4 GB cgroup — it wrongly
discarded every strong model. v2 runs each candidate INSIDE a real
``linux/amd64`` container capped exactly like the submission
(``--memory=4g --memory-swap=4g --cpus=2``): the RAM verdict is the
kernel's own OOM pass/fail, read back via ``State.OOMKilled``.

Per candidate in models/bench/<name>/model.gguf:
1. docker run the bench image with the weights mounted read-only;
   eval/bench_one.py measures load, speed and dev-set quality inside.
2. OOM under the default footprint (ctx=1536, KV q8_0) -> one retry
   with ctx=1024, KV q4_0. Still OOM -> DISCARD.
3. Apply the hard gates, write eval/benchmark_results.json +
   eval/BENCHMARK_REPORT.md (always BEFORE deletion), pick the winner,
   then install it into models/ + config.json and delete the losers.

Usage (from agent/):
    python eval/bench_models.py               # bench all + report + cleanup
    AUTO_DELETE=0 python eval/bench_models.py # stop after printing the plan
    KEEP_FALLBACK=1 ...                       # also keep the fast floor model

Requires Docker (Desktop) with linux containers. The bench image is
built automatically from eval/Dockerfile.bench when missing.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = AGENT_DIR / "eval" / "out"
BENCH_DIR = AGENT_DIR / "models" / "bench"
SHIPPED_DIR = AGENT_DIR / "models"
RESULTS_JSON = AGENT_DIR / "eval" / "benchmark_results.json"
REPORT_MD = AGENT_DIR / "eval" / "BENCHMARK_REPORT.md"

IMAGE = "track1-bench"
# The submission shape (spec §1). memory-swap == memory -> no swap escape.
DOCKER_LIMITS = ("--memory=4g", "--memory-swap=4g", "--cpus=2",
                 "--platform", "linux/amd64")
DEFAULT_FOOTPRINT = {"CTX": "1536", "KV": "q8_0"}
RETRY_FOOTPRINT = {"CTX": "1024", "KV": "q4_0"}
CONTAINER_TIMEOUT_S = 3600

# Hard gates (spec §1). RAM is the container's OOM verdict, not a threshold.
MAX_LOAD_S = 60.0
MIN_TOKENS_30S = 64
OBJECTIVE_CATEGORIES = ("math", "ner", "sentiment")
# Quality floor if the baseline container run itself fails: v1 numbers.
FALLBACK_BASELINE = {"overall_pass": 0.90,
                     "pass_by_category": {"math": 0.8, "ner": 1.0, "sentiment": 1.0}}

# Grading vCPUs are shared/slower than the bench container's (spec §8):
# ship a max_tokens_cap well inside the measured 30 s estimate.
SHIP_CAP_MARGIN = 0.6
SHIP_CAP_RANGE = (64, 200)

# name, bench subdir, needs /no_think.
CANDIDATES = (
    ("SmolLM3-3B", "smollm3-3b", True),
    ("Llama-3.2-3B-Instruct", "llama-3.2-3b", False),
    ("Qwen3.5-4B", "qwen3.5-4b", True),
    ("Phi-4-mini-instruct", "phi-4-mini", False),
    ("Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", False),
    ("Qwen2.5-3B-Instruct", "qwen2.5-3b", False),
)
BASELINE = ("Qwen2.5-0.5B", "qwen2.5-0.5b", False)

PRE_SKIPPED = (
    {"model": "Gemma-3-4B-it-QAT", "verdict": "SKIPPED",
     "reason": "excluded this round (spec §2): worst v1 RAM overflow and "
               "math regressed to 0.60"},
)


def sh(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kwargs)


def ensure_docker() -> None:
    if shutil.which("docker") is None:
        sys.exit(
            "docker not found on PATH. The v2 bench needs Docker Desktop "
            "(linux containers / WSL2). Install it, then re-run "
            "`python eval/bench_models.py` from agent/."
        )
    probe = sh(["docker", "info", "--format", "{{.OSType}}"])
    if probe.returncode != 0:
        sys.exit(f"docker daemon not reachable: {probe.stderr.strip()}")
    if probe.stdout.strip() != "linux":
        sys.exit("Docker is in Windows-containers mode; switch to Linux containers.")


def ensure_image() -> None:
    if sh(["docker", "image", "inspect", IMAGE]).returncode == 0:
        return
    print(f"Building bench image {IMAGE} (linux/amd64)...", flush=True)
    build = subprocess.run(
        ["docker", "buildx", "build", "--platform", "linux/amd64",
         "--load", "-t", IMAGE, "-f", "eval/Dockerfile.bench", "."],
        cwd=AGENT_DIR, timeout=2400,
    )
    if build.returncode != 0:
        sys.exit("bench image build failed")


def run_container(name: str, subdir: str, no_think: bool, footprint: dict) -> dict:
    """One capped container run. Returns {oom, exit_code, error}."""
    container = f"bench_{subdir.replace('.', '_')}"
    sh(["docker", "rm", "-f", container])  # stale leftovers

    model_dir = (BENCH_DIR / subdir).resolve()
    cmd = [
        "docker", "run", "--name", container, *DOCKER_LIMITS,
        "-e", "OMP_NUM_THREADS=2",
        "-e", f"CTX={footprint['CTX']}", "-e", f"KV={footprint['KV']}",
        "-v", f"{str(model_dir).replace(chr(92), '/')}:/model:ro",
        "-v", f"{str(OUT_DIR).replace(chr(92), '/')}:/out",
        IMAGE, "python", "/app/eval/bench_one.py",
        "--model", "/model/model.gguf", "--name", subdir,
    ]
    if no_think:
        cmd.append("--no-think")

    print(f"    ctx={footprint['CTX']} kv={footprint['KV']} ...", flush=True)
    try:
        proc = sh(cmd, timeout=CONTAINER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        sh(["docker", "kill", container])
        sh(["docker", "rm", "-f", container])
        return {"oom": False, "exit_code": -1,
                "error": f"bench exceeded {CONTAINER_TIMEOUT_S}s"}

    inspect = sh(["docker", "inspect", "-f", "{{.State.OOMKilled}}", container])
    oom = inspect.stdout.strip() == "true" or proc.returncode == 137
    sh(["docker", "rm", "-f", container])

    error = ""
    if proc.returncode != 0 and not oom:
        error = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
    return {"oom": oom, "exit_code": proc.returncode, "error": error}


def bench_candidate(name: str, subdir: str, no_think: bool) -> dict:
    """Run with the default footprint; retry tighter on OOM (spec §4.2)."""
    gguf = BENCH_DIR / subdir / "model.gguf"
    if not gguf.exists():
        return {"model": name, "verdict": "SKIPPED",
                "reason": f"no weights at models/bench/{subdir}/model.gguf "
                          "(download failed or repo unavailable)"}

    print(f"\n=== {name}", flush=True)
    outcome = run_container(name, subdir, no_think, DEFAULT_FOOTPRINT)
    retried = False
    if outcome["oom"]:
        print("    OOM-killed; retrying with the tight footprint", flush=True)
        retried = True
        outcome = run_container(name, subdir, no_think, RETRY_FOOTPRINT)

    record = {"model": name,
              "file_size_gb": round(gguf.stat().st_size / 1e9, 2),
              "oom_under_4g": outcome["oom"],
              "completed_batch": False}

    if outcome["oom"]:
        record["verdict"] = "DISCARD"
        record["reason"] = ("OOM under 4g (even at ctx=1024/KV q4_0)"
                            if retried else "OOM under 4g")
        return record
    if outcome["exit_code"] != 0:
        record["verdict"] = "DISCARD"
        record["reason"] = f"container failed: {outcome['error'] or outcome['exit_code']}"
        return record

    result_file = OUT_DIR / f"{subdir}.json"
    try:
        measured = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        record["verdict"] = "DISCARD"
        record["reason"] = f"no result file after run: {exc}"
        return record

    measured.update(record, model=name)
    if retried:
        measured["reason_note"] = "needed tight footprint (ctx=1024/KV q4_0)"
    return measured


def apply_gates(record: dict, baseline: dict) -> None:
    if record.get("verdict") in ("SKIPPED", "DISCARD"):
        return

    reasons = []
    if record["load_s"] >= MAX_LOAD_S:
        reasons.append(f"load {record['load_s']}s >= {MAX_LOAD_S}s")
    if record["est_tokens_30s"] < MIN_TOKENS_30S:
        reasons.append(
            f"only ~{record['est_tokens_30s']} tokens fit in 30s (< {MIN_TOKENS_30S})"
        )

    if record is not baseline:
        base = baseline if baseline and "overall_pass" in baseline else FALLBACK_BASELINE
        if record["overall_pass"] < base["overall_pass"]:
            reasons.append(
                f"overall {record['overall_pass']:.2f} below baseline "
                f"{base['overall_pass']:.2f}"
            )
        regressed = [
            cat for cat in OBJECTIVE_CATEGORIES
            if record["pass_by_category"].get(cat, 0)
            < base["pass_by_category"].get(cat, 0)
        ]
        if regressed:
            reasons.append("regresses " + "/".join(regressed) + " vs baseline")

    if record is baseline:
        record["verdict"] = "BASELINE" if not reasons else "DISCARD"
    else:
        record["verdict"] = "KEEP" if not reasons else "DISCARD"
    record["reason"] = "; ".join(reasons)
    if record.get("reason_note"):
        record["reason"] = "; ".join(filter(None, [record["reason"],
                                                   record.pop("reason_note")]))


def pick_winner(records: list) -> tuple:
    """(winner, forced) per spec §5; forced=True when no candidate KEEPs."""
    keepers = [r for r in records if r.get("verdict") == "KEEP"]
    if keepers:
        keepers.sort(key=lambda r: (-r["overall_pass"], -r["decode_tok_s"],
                                    r["file_size_gb"]))
        return keepers[0], False

    survivors = [r for r in records
                 if r.get("completed_batch") and not r.get("oom_under_4g")]
    if not survivors:
        return None, True
    survivors.sort(key=lambda r: -r["overall_pass"])
    return survivors[0], True


def write_report(records: list, winner: dict, forced: bool) -> None:
    RESULTS_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    def cell(r, key, fmt="{}"):
        return fmt.format(r[key]) if key in r else "—"

    lines = [
        "# Local model benchmark v2 — container-based (Track 1)",
        "",
        "Each candidate ran inside a `linux/amd64` container capped at "
        "`--memory=4g --memory-swap=4g --cpus=2` — the submission shape — so "
        "the RAM verdict is the kernel's OOM pass/fail (`State.OOMKilled`), "
        "not a host-RSS guess (v1's mistake). Footprint: ctx 1536 + q8_0 KV "
        "cache by default, one OOM retry at ctx 1024 + q4_0. Quality = "
        "offline heuristic judge over the 30 locally-served dev tasks; code "
        "categories route to Fireworks in production. Speed here is still "
        "optimistic vs the shared grading vCPUs — the shipped token cap "
        f"takes a {SHIP_CAP_MARGIN:.0%} margin on the 30 s estimate.",
        "",
        "| model | size(GB) | load(s) | OOM@4g | cgroup peak(GB) | ctx/KV | "
        "decode tok/s | est tok≤30s | overall | math | ner | sentiment | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(
        records,
        key=lambda r: (r.get("verdict") == "SKIPPED", -r.get("overall_pass", -1)),
    )
    for r in ranked:
        cats = r.get("pass_by_category", {})
        ctx_kv = f"{r['ctx']}/{r['kv']}" if "ctx" in r else "—"
        lines.append(
            f"| {r['model']} | {cell(r, 'file_size_gb')} | {cell(r, 'load_s')} | "
            f"{cell(r, 'oom_under_4g')} | {cell(r, 'cgroup_peak_gb')} | {ctx_kv} | "
            f"{cell(r, 'decode_tok_s')} | {cell(r, 'est_tokens_30s')} | "
            f"{cell(r, 'overall_pass')} | {cats.get('math', '—')} | "
            f"{cats.get('ner', '—')} | {cats.get('sentiment', '—')} | "
            f"{r.get('verdict', '?')}{': ' + r['reason'] if r.get('reason') else ''} |"
        )

    lines.append("")
    if winner is None:
        lines.append("**No candidate survived the 4 GB container — investigate before shipping.**")
    else:
        lines.append(
            f"**Winner: {winner['model']}** — overall pass "
            f"{winner['overall_pass']:.2f}, {winner['decode_tok_s']} decode tok/s "
            f"in-container, {winner['file_size_gb']} GB, survived the full batch "
            f"under 4 GB (cgroup peak {winner.get('cgroup_peak_gb', '?')} GB, "
            f"ctx {winner.get('ctx')}/{winner.get('kv')} KV)."
        )
        if forced:
            lines.append(
                "**WARNING: no candidate passed every hard gate.** This is the "
                "best model that survives 4 GB; the 30 s limit forces a lower "
                "`local_max_tokens_cap`."
            )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {RESULTS_JSON} and {REPORT_MD}")


def ship_cap_from(record: dict) -> int:
    lo, hi = SHIP_CAP_RANGE
    return max(lo, min(hi, int(record["est_tokens_30s"] * SHIP_CAP_MARGIN)))


def cleanup(records: list, winner: dict) -> None:
    """Spec §7: plan, install winner into models/ + config.json, delete losers."""
    by_model = {r["model"]: r for r in records}
    subdir_of = {name: subdir for name, subdir, _ in (*CANDIDATES, BASELINE)}

    fallback = None
    if os.environ.get("KEEP_FALLBACK") == "1":
        floors = [r for r in records
                  if r.get("completed_batch") and not r.get("oom_under_4g")
                  and r is not winner]
        if floors:
            fallback = min(floors, key=lambda r: r["file_size_gb"])

    kept_records = [winner] + ([fallback] if fallback else [])
    kept_sources = {}
    for r in kept_records:
        src = (BENCH_DIR / subdir_of[r["model"]] / "model.gguf").resolve()
        kept_sources[src] = (SHIPPED_DIR / f"{subdir_of[r['model']]}.gguf").resolve()

    delete_paths = [
        Path(f).resolve()
        for pattern in (str(BENCH_DIR / "*" / "*.gguf"), str(SHIPPED_DIR / "*.gguf"))
        for f in glob.glob(pattern)
        if Path(f).resolve() not in kept_sources
        and Path(f).resolve() not in kept_sources.values()
    ]
    freed = sum(p.stat().st_size for p in delete_paths)

    print("\n=== Cleanup plan ===")
    for src, dst in kept_sources.items():
        print(f"KEEP: {src} -> {dst}")
    for p in delete_paths:
        print(f"DELETE ({p.stat().st_size / 1e9:.2f} GB): {p}")
    print(f"Would free ~{freed / 1e9:.2f} GB")

    if os.environ.get("AUTO_DELETE", "1") == "0":
        print("AUTO_DELETE=0 -> stopping after the plan; nothing deleted.")
        return

    # Copy kept weights into the shipped dir BEFORE deleting anything.
    for src, dst in kept_sources.items():
        if not dst.exists():
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)

    winner_file = kept_sources[
        (BENCH_DIR / subdir_of[winner["model"]] / "model.gguf").resolve()
    ]
    config_path = AGENT_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["local_model_path"] = f"models/{winner_file.name}"
    config["local_ctx"] = winner.get("ctx", 1536)
    config["local_kv_type"] = winner.get("kv", "q8_0")
    config["local_max_tokens_cap"] = ship_cap_from(winner)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"config.json: local_model_path=models/{winner_file.name} "
          f"local_ctx={config['local_ctx']} local_kv_type={config['local_kv_type']} "
          f"local_max_tokens_cap={config['local_max_tokens_cap']}")

    for p in delete_paths:
        p.unlink()
    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
    print(f"Freed ~{freed / 1e9:.2f} GB")
    print("Final shipped model(s): "
          + ", ".join(str(d) for d in kept_sources.values()))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ensure_docker()
    ensure_image()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = bench_candidate(*BASELINE)
    records = [bench_candidate(name, subdir, no_think)
               for name, subdir, no_think in CANDIDATES]
    records.append(baseline)

    apply_gates(baseline, baseline)
    for record in records:
        if record is not baseline:
            apply_gates(record, baseline)
    records.extend(PRE_SKIPPED)

    winner, forced = pick_winner(records)
    write_report(records, winner, forced)

    if winner is None:
        print("\nNO WINNER — nothing deleted.")
        return 1

    survived = "yes" if not winner.get("oom_under_4g") else "no"
    print(f"\nWINNER: {winner['model']} (pass={winner['overall_pass']:.2f}, "
          f"decode={winner['decode_tok_s']} tok/s, "
          f"size={winner['file_size_gb']} GB, survived 4g={survived})")

    cleanup(records, winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
