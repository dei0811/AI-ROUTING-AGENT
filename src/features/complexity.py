"""Complexity feature extraction module.

Estimates heuristic complexity indicators for a prompt using rule-based
counting and pattern matching only. No AI models are used and no final
composite complexity score is computed -- only raw features.
"""

import re

_REASONING_KEYWORDS_RE = re.compile(
    r"\b(reason\w*|think\w* (through|about)|analy\w*|deduc\w*|infer\w*|"
    r"raz\w*|piensa|analiza)\b",
    re.IGNORECASE,
)
_STEP_BY_STEP_RE = re.compile(
    r"\bstep[- ]by[- ]step\b|\bpaso a paso\b", re.IGNORECASE
)
_CHAIN_OF_THOUGHT_RE = re.compile(
    r"\bchain of thought\b|\bthink step by step\b|\bshow your (reasoning|work)\b|"
    r"\brazona\w* paso a paso\b|\bmuestra tu razonamiento\b",
    re.IGNORECASE,
)
_MATH_SYMBOLS_RE = re.compile(r"[+\-*/^=<>%√∑∫π]|\\frac|\\sum|\\int")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_LOGICAL_CONNECTORS_RE = re.compile(
    r"\b(however|therefore|moreover|furthermore|although|because|"
    r"since|thus|hence|nevertheless|sin embargo|por lo tanto|adem[aá]s|"
    r"aunque|porque|ya que|por consiguiente|no obstante)\b",
    re.IGNORECASE,
)
_CONSTRAINT_KEYWORDS_RE = re.compile(
    r"\b(must|should|require\w*|at least|at most|no more than|only|"
    r"exactly|debe\w*|requiere\w*|al menos|como m[aá]ximo|no m[aá]s de|"
    r"solamente|exactamente)\b",
    re.IGNORECASE,
)
_NUMERIC_CONSTRAINT_RE = re.compile(
    r"\b(at least|at most|no more than|exactly|al menos|como m[aá]ximo|"
    r"exactamente)\s+\d+|\b\d+\s*(words|characters|palabras|caracteres|"
    r"lines|l[ií]neas)\b",
    re.IGNORECASE,
)
_INSTRUCTION_VERBS_RE = re.compile(
    r"^\s*(write|create|explain|generate|translate|analyze|summarize|"
    r"list|compare|solve|implement|design|build|describe|escribe|crea|"
    r"explica|genera|traduce|analiza|resume|lista|compara|resuelve|"
    r"implementa|dise[nñ]a|construye|describe)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _count_questions(text: str) -> int:
    """Count the number of question marks as a proxy for question count."""
    return text.count("?")


def _max_parenthesis_depth(text: str) -> int:
    """Compute the maximum nesting depth of parentheses in the text."""
    depth = 0
    max_depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def _contains_nested_lists(text: str) -> bool:
    """Detect nested list structures via indented bullet/numbered items."""
    lines = text.splitlines()
    indented_bullet = re.compile(r"^(\s{2,}|\t+)([-*+]|\d+[.)])\s+\S")
    return any(indented_bullet.match(line) for line in lines)


def _average_sentence_complexity(text: str) -> float:
    """Approximate sentence complexity as average words per sentence."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    total_words = sum(len(s.split()) for s in sentences)
    return round(total_words / len(sentences), 4)


def extract_complexity(prompt: str) -> dict:
    """Extract heuristic complexity features from a prompt.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A dictionary of complexity-related features computed purely
        through rule-based heuristics. No composite score is returned.
    """
    text = prompt if isinstance(prompt, str) else ""

    prompt_length = len(text)
    context_length = len(text)
    estimated_token_count = max(1, len(text) // 4) if text else 0

    num_questions = _count_questions(text)
    multiple_questions = num_questions > 1

    reasoning_keywords = len(_REASONING_KEYWORDS_RE.findall(text))
    constraint_count = len(_CONSTRAINT_KEYWORDS_RE.findall(text))
    numeric_constraints = len(_NUMERIC_CONSTRAINT_RE.findall(text))
    contains_step_by_step = bool(_STEP_BY_STEP_RE.search(text))
    contains_chain_of_thought_words = bool(_CHAIN_OF_THOUGHT_RE.search(text))
    contains_math_symbols = bool(_MATH_SYMBOLS_RE.search(text))
    contains_nested_lists = _contains_nested_lists(text)

    instruction_matches = _INSTRUCTION_VERBS_RE.findall(text)
    instruction_count = len(instruction_matches)
    contains_multiple_tasks = instruction_count > 1

    code_block_count = len(_CODE_BLOCK_RE.findall(text))
    quotation_count = text.count('"') // 2 + text.count("'") // 2
    parenthesis_depth = _max_parenthesis_depth(text)
    average_sentence_complexity = _average_sentence_complexity(text)
    logical_connector_count = len(_LOGICAL_CONNECTORS_RE.findall(text))

    return {
        "prompt_length": prompt_length,
        "context_length": context_length,
        "estimated_token_count": estimated_token_count,
        "multiple_questions": multiple_questions,
        "reasoning_keywords": reasoning_keywords,
        "constraint_count": constraint_count,
        "numeric_constraints": numeric_constraints,
        "contains_step_by_step": contains_step_by_step,
        "contains_chain_of_thought_words": contains_chain_of_thought_words,
        "contains_math_symbols": contains_math_symbols,
        "contains_nested_lists": contains_nested_lists,
        "contains_multiple_tasks": contains_multiple_tasks,
        "code_block_count": code_block_count,
        "quotation_count": quotation_count,
        "parenthesis_depth": parenthesis_depth,
        "average_sentence_complexity": average_sentence_complexity,
        "logical_connector_count": logical_connector_count,
        "instruction_count": instruction_count,
    }
