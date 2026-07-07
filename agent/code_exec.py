"""Safe local runner for LLM-generated Python code.

Executes the code in a fresh subprocess with a hard timeout, isolated
mode (-I: no env vars, no user site-packages), and captured output.
Local execution is free (spec: local compute is allowed for executing
LLM-generated code), so paying a few completion tokens for a short
program instead of long chain-of-thought is a net token win.
"""

import logging
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 6.0
_MAX_OUTPUT_CHARS = 4000

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull Python code out of a completion.

    Takes the fenced ```python block(s) if present (joined in order),
    otherwise assumes the whole completion is code.
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.strip()


def run_python(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple:
    """Run Python source in an isolated subprocess.

    Args:
        code: Python source to execute.
        timeout_s: Hard wall-clock limit; the process is killed after it.

    Returns:
        (ok, output): ok is True only if the process exited 0 AND printed
        something on stdout. output is the stripped stdout on success,
        or a short diagnostic (stderr tail / "timeout") on failure.
    """
    if not code.strip():
        return False, "empty code"

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # e.g. OSError spawning the interpreter
        return False, str(exc)

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-500:]
        return False, stderr_tail or f"exit code {proc.returncode}"

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return False, "no output"

    return True, stdout[:_MAX_OUTPUT_CHARS]
