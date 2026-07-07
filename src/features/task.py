"""Task-type feature extraction module.

Detects candidate task types implied by a prompt using only keyword
lists and regular expressions (no machine learning models). Multiple
task flags can be active simultaneously since a prompt may request
several things at once. Keyword lists include both English and Spanish
terms for broader coverage.
"""

import re

_TASK_KEYWORDS = {
    "task_translation": [
        r"\btranslat\w*", r"\btraduc\w*", r"\bin (english|spanish|french|german)\b",
        r"\bal (ingl[eé]s|espa[nñ]ol|franc[eé]s|alem[aá]n)\b",
    ],
    "task_programming": [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bprogram\w*",
        r"\bc[oó]digo\b", r"\bfunci[oó]n\b", r"\balgorithm\w*", r"\balgoritmo\b",
        r"\bpython\b", r"\bjavascript\b", r"\bjava\b", r"\bc\+\+\b",
    ],
    "task_math": [
        r"\bcalculat\w*", r"\bsolve\w*", r"\bequation\w*", r"\bderivative\w*",
        r"\bintegral\w*", r"\bcalcul\w*", r"\bresolv\w*", r"\bresuelv\w*",
        r"\becuaci[oó]n\w*", r"\bderivada\w*", r"\bmatem[aá]tica\w*",
    ],
    "task_reasoning": [
        r"\bwhy\b", r"\breason\w*", r"\blogic\w*", r"\bpor qu[eé]\b",
        r"\braz[oó]n\w*", r"\bl[oó]gica\w*", r"\bdeduc\w*", r"\binfer\w*",
    ],
    "task_explanation": [
        r"\bexplain\w*", r"\bexplic\w*", r"\bexpliqu\w*", r"\bwhat is\b", r"\bqu[eé] es\b",
        r"\bhow does\b", r"\bc[oó]mo funciona\b",
    ],
    "task_summary": [
        r"\bsummariz\w*", r"\bsummary\b", r"\bresum\w*", r"\btl;?dr\b",
    ],
    "task_generation": [
        r"\bgenerat\w*", r"\bcreate\b", r"\bgener\w*", r"\bcrea\w*",
        r"\bwrite (a|an|me)\b", r"\bescribe\b", r"\bredacta\w*",
    ],
    "task_classification": [
        r"\bclassif\w*", r"\bcategoriz\w*", r"\bclasific\w*",
        r"\blabel\w* this\b", r"\betiqueta\w*",
    ],
    "task_analysis": [
        r"\banaly\w*", r"\banaliz\w*", r"\bevaluat\w*", r"\beval[uú]a\w*",
    ],
    "task_debugging": [
        r"\bdebug\w*", r"\bfix\w* (this|the|my)?\s*(bug|error|code)\b",
        r"\berror\b", r"\bexception\b", r"\btraceback\b",
        r"\bdepura\w*", r"\bcorrig\w*", r"\bcorreg\w*",
    ],
    "task_question_answering": [
        r"\?\s*$", r"^\s*(what|who|when|where|which|how)\b",
        r"^\s*(qu[eé]|qui[eé]n|cu[aá]ndo|d[oó]nde|cu[aá]l|c[oó]mo)\b",
    ],
    "task_rewriting": [
        r"\brewrit\w*", r"\brephras\w*", r"\bparaphras\w*",
        r"\breescrib\w*", r"\breformul\w*", r"\bpar[aá]frase\w*",
    ],
    "task_planning": [
        r"\bplan\w*", r"\bschedule\w*", r"\bstrategy\b", r"\bstrateg\w*",
        r"\bplanifica\w*", r"\bplanea\w*",
    ],
    "task_comparison": [
        r"\bcompar\w*", r"\bversus\b", r"\bvs\.?\b", r"\bdifference\w* between\b",
        r"\bdiferencia\w* entre\b",
    ],
    "task_brainstorming": [
        r"\bbrainstorm\w*", r"\bideas?\b", r"\bideas? para\b", r"\bideas? for\b",
        r"\bpropon\w*", r"\bsugier\w*", r"\bsuger\w*",
    ],
    "task_roleplay": [
        r"\bact as\b", r"\bpretend (you are|to be)\b", r"\bact[uú]a como\b",
        r"\bfinge que\b", r"\brole[- ]?play\b", r"\bimagina que eres\b",
    ],
    "task_code_generation": [
        r"\bwrite (a|the)? ?(function|script|program|code)\b",
        r"\bimplement\w*", r"\bimplementa\w*",
        r"\bescribe (una|un) (funci[oó]n|script|programa|c[oó]digo)\b",
    ],
    "task_data_analysis": [
        r"\bdataset\b", r"\bdataframe\b", r"\bdata analysis\b",
        r"\banalisis de datos\b", r"\ban[aá]lisis de datos\b",
        r"\bcsv\b", r"\bpandas\b", r"\bestad[ií]stic\w*",
    ],
    "task_extraction": [
        r"\bextract\w*", r"\bextrae\w*", r"\bparse\b", r"\bpull out\b",
    ],
    "task_search": [
        r"\bsearch\w*", r"\bfind\w* (information|out)\b", r"\bbusca\w*",
        r"\bencuentra\w*", r"\bencontr\w*", r"\blook up\b",
    ],
}

_COMPILED_KEYWORDS = {
    task: [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
    for task, patterns in _TASK_KEYWORDS.items()
}


def extract_task(prompt: str) -> dict:
    """Detect candidate task types present in a prompt.

    Uses only keyword lists and regular expressions (no LLM calls).
    Multiple task flags can be True simultaneously.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A dictionary mapping each ``task_*`` feature name to a boolean
        indicating whether that task type appears to be present.
    """
    text = prompt if isinstance(prompt, str) else ""

    features = {}
    for task_name, patterns in _COMPILED_KEYWORDS.items():
        features[task_name] = any(pattern.search(text) for pattern in patterns)

    return features
