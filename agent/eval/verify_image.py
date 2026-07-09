"""Verify the shipping image against every hard rule, under the real caps.

Runs track1-agent:verify on the 8 practice tasks inside a container
capped exactly like the grading box (--memory=4g --memory-swap=4g
--cpus=2, linux/amd64) with MOCK_FIREWORKS=1 (no credentials yet) and
BENCH_TIMING=1, then asserts:

  RAM        State.OOMKilled == false
  EXIT       container exit code 0
  STARTUP    STARTUP_DONE within 60 s of launch
  PER_TASK   every TASK line < 30 s (warn > 24 s: grading vCPUs are slower)
  BATCH      total wall time < 10 min
  SCHEMA     /output/results.json valid; practice-01..08 exactly once,
             each with a non-empty string answer
  ENGLISH    answers look English (ASCII-letter heuristic)
  WEIGHTS    image contains exactly one GGUF (the winner)
  SIZE       docker-save + gzip size <= 10 GB (proxy for registry size)

Prints one PASS/FAIL line per rule; exits non-zero on any FAIL.

Usage (from agent/):
    python eval/verify_image.py [--image track1-agent:verify] [--keep]
"""

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
PRACTICE_DIR = AGENT_DIR / "practice"
OUT_DIR = AGENT_DIR / "out"

CONTAINER = "t1_verify"
EXPECTED_IDS = [f"practice-{i:02d}" for i in range(1, 9)]

STARTUP_LIMIT_S = 60.0
TASK_LIMIT_S = 30.0
TASK_WARN_S = 24.0
BATCH_LIMIT_S = 600.0
SIZE_LIMIT_BYTES = 10 * 1024**3

RUN_TIMEOUT_S = 720  # docker-run safety net beyond the 10-min rule

_STARTUP_RE = re.compile(r"^STARTUP_DONE ([0-9.]+)", re.MULTILINE)
_TASK_RE = re.compile(r"^TASK (\S+) ([0-9.]+)", re.MULTILINE)


def sh(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kwargs)


class Verifier:
    def __init__(self):
        self.failed = False

    def check(self, rule: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {rule:<9} {detail}")
        if not ok:
            self.failed = True


def looks_english(text: str) -> bool:
    """Lenient heuristic: the letters used are overwhelmingly ASCII.

    Code/JSON answers pass trivially; non-Latin scripts fail.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True  # numbers/JSON-only answers carry no language signal
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= 0.9


def compressed_image_size(image: str) -> int:
    """Approximate registry-compressed size: docker save | gzip, counted."""
    proc = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    total = 0

    class _Counter:
        def write(self, data):
            nonlocal total
            total += len(data)
            return len(data)

    gz = gzip.GzipFile(fileobj=_Counter(), mode="wb", compresslevel=6)
    while True:
        chunk = proc.stdout.read(1024 * 1024)
        if not chunk:
            break
        gz.write(chunk)
    gz.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("docker save failed")
    return total


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="track1-agent:verify")
    parser.add_argument("--keep", action="store_true",
                        help="keep the container and /out for inspection")
    args = parser.parse_args()

    tasks_file = PRACTICE_DIR / "tasks.json"
    if not tasks_file.exists():
        sys.exit(f"missing {tasks_file} — create the 8 practice tasks first")
    OUT_DIR.mkdir(exist_ok=True)
    results_file = OUT_DIR / "results.json"
    results_file.unlink(missing_ok=True)

    v = Verifier()
    sh(["docker", "rm", "-f", CONTAINER])

    run_cmd = [
        "docker", "run", "--name", CONTAINER, "--platform", "linux/amd64",
        "--memory=4g", "--memory-swap=4g", "--cpus=2",
        "-e", "MOCK_FIREWORKS=1", "-e", "BENCH_TIMING=1",
        # Harness-injected on the grading box; dummies keep mock mode honest.
        "-e", "FIREWORKS_API_KEY=mock",
        "-e", "FIREWORKS_BASE_URL=http://mock.invalid",
        "-e", "ALLOWED_MODELS=mock-small-2b,mock-large-70b",
        "-v", f"{str(PRACTICE_DIR).replace(chr(92), '/')}:/input:ro",
        "-v", f"{str(OUT_DIR).replace(chr(92), '/')}:/output",
        args.image,
    ]
    print(f"Running {args.image} under 4g/2cpu on the practice tasks...\n")
    t0 = time.time()
    try:
        proc = sh(run_cmd, timeout=RUN_TIMEOUT_S)
        wall_s = time.time() - t0
        timed_out = False
    except subprocess.TimeoutExpired:
        sh(["docker", "kill", CONTAINER])
        wall_s = time.time() - t0
        timed_out = True
        proc = None

    oom = sh(["docker", "inspect", "-f", "{{.State.OOMKilled}}", CONTAINER]
             ).stdout.strip() == "true"
    logs = "" if proc is None else (proc.stdout or "") + (proc.stderr or "")

    # --- hard rules
    v.check("RAM", not oom, "State.OOMKilled == false" if not oom
            else "container was OOM-killed under --memory=4g")
    exit_ok = proc is not None and proc.returncode == 0 and not timed_out
    v.check("EXIT", exit_ok,
            f"exit code {'timeout' if timed_out else proc.returncode if proc else '?'}")

    startup = _STARTUP_RE.search(logs)
    if startup:
        startup_s = float(startup.group(1)) - t0
        v.check("STARTUP", 0 <= startup_s < STARTUP_LIMIT_S,
                f"model loaded + reading /input after {startup_s:.1f}s")
    else:
        v.check("STARTUP", False, "no STARTUP_DONE line (BENCH_TIMING wiring?)")

    task_times = [(m.group(1), float(m.group(2))) for m in _TASK_RE.finditer(logs)]
    if task_times:
        worst_id, worst = max(task_times, key=lambda t: t[1])
        ok = worst < TASK_LIMIT_S
        note = f"worst task {worst_id} = {worst:.1f}s (n={len(task_times)})"
        if ok and worst > TASK_WARN_S:
            note += f"  WARNING: >{TASK_WARN_S:.0f}s leaves little margin for the slower grading VM"
        v.check("PER_TASK", ok, note)
    else:
        v.check("PER_TASK", False, "no TASK timing lines found")

    v.check("BATCH", wall_s < BATCH_LIMIT_S, f"batch wall time {wall_s:.0f}s")

    # --- results schema
    schema_ok = False
    answers = {}
    try:
        results = json.loads(results_file.read_text(encoding="utf-8"))
        ids = [e.get("task_id") for e in results]
        schema_ok = (
            isinstance(results, list)
            and sorted(ids) == sorted(EXPECTED_IDS)
            and len(ids) == len(set(ids))
            and all(isinstance(e.get("answer"), str) and e["answer"].strip()
                    for e in results)
        )
        answers = {e.get("task_id"): e.get("answer", "") for e in results}
    except (OSError, ValueError) as exc:
        v.check("SCHEMA", False, f"results.json unreadable: {exc}")
    else:
        v.check("SCHEMA", schema_ok,
                "all 8 ids exactly once, non-empty string answers" if schema_ok
                else f"ids={ids} empties="
                     f"{[i for i, a in answers.items() if not str(a).strip()]}")

    non_english = [tid for tid, ans in answers.items()
                   if not looks_english(str(ans))]
    v.check("ENGLISH", not non_english,
            "all answers look English" if not non_english
            else f"suspect answers: {non_english}")

    # --- image contents + size
    ls = sh(["docker", "run", "--rm", "--entrypoint", "ls", args.image,
             "/app/models"])
    ggufs = [l for l in ls.stdout.split() if l.endswith(".gguf")]
    v.check("WEIGHTS", ls.returncode == 0 and len(ggufs) == 1,
            f"/app/models: {ggufs}")

    try:
        size = compressed_image_size(args.image)
        v.check("SIZE", size <= SIZE_LIMIT_BYTES,
                f"~{size / 1024**3:.2f} GiB gzip-compressed (proxy for registry size)")
    except Exception as exc:
        v.check("SIZE", False, f"could not measure: {exc}")

    if not args.keep:
        sh(["docker", "rm", "-f", CONTAINER])

    print(f"\n{'ALL RULES PASS' if not v.failed else 'VERIFICATION FAILED'}")
    return 1 if v.failed else 0


if __name__ == "__main__":
    sys.exit(main())
