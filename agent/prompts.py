"""Terse per-category prompt templates.

Every character of a system prompt is paid on every call, so these are
as short as accuracy allows (spec §7: no personas, no few-shot unless a
category fails without it). Tune wording on launch day via local eval.

Output constraints per category follow the spec §6 table: direct
answers, labels only, compact JSON for NER, code only for code tasks.
"""

from classify import (
    CODE_DEBUG,
    CODE_GEN,
    FACTUAL,
    LOGIC,
    MATH,
    NER,
    SENTIMENT,
    SUMMARIZATION,
    UNKNOWN,
)

GENERAL_SYSTEM_PROMPT = (
    "Answer in English. Be direct and brief. "
    "No preamble. Do not restate the question."
)

# Emit-code path (math, optionally logic): the program runs locally for
# free, replacing paid chain-of-thought tokens with a short code block.
CODE_EMIT_SYSTEM_PROMPT = (
    "Write a Python 3 program that computes the answer and prints only "
    "the final answer. Output only code."
)

SYSTEM_PROMPTS = {
    FACTUAL: "Answer in English with only the requested fact. No explanation.",
    # Phase 5 replaces direct math answers with emit-code -> local execution.
    MATH: "Answer with only the final numeric result. No steps.",
    SENTIMENT: "Reply with exactly one word: positive, negative, or neutral.",
    # Terse: summarization is prefill-bound on CPU, every input token
    # costs latency. The user prompt carries the length request.
    SUMMARIZATION: "Output only the summary, in English.",
    NER: "Extract the requested entities. Output compact JSON only. No prose.",
    CODE_DEBUG: "Output only the corrected code. No explanation.",
    LOGIC: "Answer in English with only the final answer. No reasoning steps.",
    CODE_GEN: "Output only the code. No explanation.",
    UNKNOWN: GENERAL_SYSTEM_PROMPT,
}

# Fallback output caps; config.json overrides win. Minimums that should
# still pass the judge — trim further on launch day.
DEFAULT_MAX_TOKENS = {
    FACTUAL: 64,
    MATH: 64,
    SENTIMENT: 8,
    SUMMARIZATION: 256,
    NER: 256,
    CODE_DEBUG: 512,
    LOGIC: 128,
    CODE_GEN: 512,
    UNKNOWN: 256,
}


def build_messages(category: str, prompt: str) -> list:
    """Build the minimal message list for a task.

    Args:
        category: One of classify.CATEGORIES.
        prompt: The raw task prompt (sent verbatim; never restated).

    Returns:
        OpenAI-style messages: terse category system prompt + the task.
    """
    system = SYSTEM_PROMPTS.get(category, GENERAL_SYSTEM_PROMPT)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
