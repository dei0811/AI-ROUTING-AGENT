"""Local heuristic category detection. Token-free by design.

Maps a prompt to one of the 8 scored capability categories (or
"unknown"). Plain regex/keyword heuristics only — no models — so
classification costs zero tokens and no Fireworks calls (spec §5).

Categories are checked from most-specific cues to most-generic:
explicit task verbs (summarise, extract entities...) win over generic
question shapes. "unknown" falls back to the general prompt in solve.
"""

import re

FACTUAL = "factual"
MATH = "math"
SENTIMENT = "sentiment"
SUMMARIZATION = "summarization"
NER = "ner"
CODE_DEBUG = "code_debug"
LOGIC = "logic"
CODE_GEN = "code_gen"
UNKNOWN = "unknown"

CATEGORIES = (
    FACTUAL, MATH, SENTIMENT, SUMMARIZATION,
    NER, CODE_DEBUG, LOGIC, CODE_GEN, UNKNOWN,
)

_CODE_PRESENT_RE = re.compile(
    r"```|(?:^|\n)\s*(?:def |class |import |from \w+ import |function |const |var |let )"
    r"|;\s*\n|\{\s*\n",
)

_SENTIMENT_RE = re.compile(
    r"\bsentiment\b"
    r"|positive[,/ ]+(or )?negative"
    r"|\bclassify\b.*\b(review|tweet|opinion|feedback|text)\b"
    r"|\b(review|tweet|opinion|feedback)\b.*\bclassify\b"
    r"|is (this|the following) (review|tweet|text|comment) (positive|negative)",
    re.IGNORECASE | re.DOTALL,
)

_SUMMARIZATION_RE = re.compile(
    r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b"
    r"|\b(condense|shorten)\b"
    r"|\bin (one|a single|\d+) sentences?\b"
    r"|\bmain (idea|points?)\b",
    re.IGNORECASE,
)

_NER_RE = re.compile(
    r"named entit|\bentit(y|ies)\b|\bNER\b"
    r"|\b(extract|identify|find|list)\b.{0,40}\b"
    r"(people|persons?|names?|organi[sz]ations?|companies|locations?|places|dates?)\b",
    re.IGNORECASE | re.DOTALL,
)

_DEBUG_KEYWORDS_RE = re.compile(
    r"\b(fix|debug|bug|error|broken|not work\w*|doesn'?t work|incorrect|wrong"
    r"|traceback|exception|crash\w*|throws?|rais(e|es|ed|ing)|fail(s|ed|ing)?)\b"
    r"|\w+(Error|Exception)\b",
    re.IGNORECASE,
)

_CODE_GEN_RE = re.compile(
    r"\b(write|create|implement|generate|build|develop)\b.{0,60}\b"
    r"(function|program|script|class|method|code|algorithm|regex|query|api|app)\b"
    r"|\bcode\b.{0,20}\b(that|to|which)\b"
    r"|\bin (python|javascript|java|c\+\+|c#|go|rust|sql|typescript)\b",
    re.IGNORECASE | re.DOTALL,
)

_MATH_RE = re.compile(
    r"\d\s*[\+\-\*/×÷^%]\s*\d"
    r"|\b(calculate|compute|solve)\b"
    r"|\bhow (much|many)\b"
    r"|\b(sum|product|difference|quotient|average|mean|median|percent(age)?"
    r"|remainder|equation|integral|derivative|probability|fraction)\b"
    r"|\d\s*%"
    r"|=\s*\?",
    re.IGNORECASE,
)

_LOGIC_RE = re.compile(
    r"\b(deduce|deduct\w*|infer|premise|syllogism|conclusion|logically"
    r"|logic puzzle|riddle|paradox)\b"
    r"|\ball \w+ are \w+"
    r"|\bif\b.{3,80}\bthen\b"
    r"|\bthan\b.{0,60}\bthan\b"
    r"|\bwho (is|are)\b.{0,40}\b((tall|old|young|short|fast|slow)(er|est)|left|right)\b"
    # Assignment puzzles: "X, Y and Z each own exactly one pet ...
    # Which pet does each person own?"
    r"|\beach\b.{0,40}\b(owns?|has|have|gets?|likes?)\b.{0,20}\bone\b"
    r"|\b(which|what|who)\b[^.?]{0,60}\bdoes each\b"
    # Propositional disjunction: "Either it is raining or it is sunny."
    r"|\beither\b.{3,80}\bor\b"
    r"|true or false",
    re.IGNORECASE | re.DOTALL,
)

_FACTUAL_RE = re.compile(
    r"^(who|what|when|where|which|why|how)\b"
    r"|\b(capital|president|inventor|author|currency|population|located|founded"
    r"|discovered|largest|smallest|highest|longest)\b"
    r"|\?\s*$",
    re.IGNORECASE,
)

# Most-specific first: explicit task verbs beat generic question shapes.
# Math/logic come after the code categories so "write a function to sum..."
# lands in code_gen, and before factual so "how many..." lands in math.
_RULES = (
    (SENTIMENT, _SENTIMENT_RE),
    (NER, _NER_RE),
    (SUMMARIZATION, _SUMMARIZATION_RE),
    (CODE_GEN, _CODE_GEN_RE),
    (MATH, _MATH_RE),
    (LOGIC, _LOGIC_RE),
    (FACTUAL, _FACTUAL_RE),
)


def classify(prompt: str) -> str:
    """Classify a prompt into one of the 8 categories or "unknown".

    Args:
        prompt: Raw task prompt.

    Returns:
        Category name (see CATEGORIES).
    """
    if not prompt or not prompt.strip():
        return UNKNOWN

    # Code present + brokenness cues → debugging, regardless of phrasing.
    has_code = bool(_CODE_PRESENT_RE.search(prompt))
    if has_code and _DEBUG_KEYWORDS_RE.search(prompt):
        return CODE_DEBUG

    for category, pattern in _RULES:
        if pattern.search(prompt):
            return category

    # Code present without debug cues usually means "complete/extend this".
    if has_code:
        return CODE_GEN

    return UNKNOWN
