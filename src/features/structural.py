"""Structural feature extraction module.

Detects structural patterns in a prompt (code blocks, markup, URLs,
emails, tables, file paths, shell commands, lists, etc.) using regular
expressions only. All returned features are booleans.
"""

import re

_CODE_BLOCK_RE = re.compile(r"```|~~~|(?:^|\n)(?: {4}|\t)\S")
_JSON_RE = re.compile(r"\{[^{}]*\"[^\"]+\"\s*:\s*.+?\}", re.DOTALL)
_XML_RE = re.compile(r"<\?xml.*?\?>|<[a-zA-Z][\w\-]*(\s+[^<>]*)?>.*?</[a-zA-Z][\w\-]*>", re.DOTALL)
_HTML_RE = re.compile(
    r"<(html|head|body|div|span|p|a|img|table|tr|td|ul|ol|li|script|style|h[1-6])\b",
    re.IGNORECASE,
)
_SQL_RE = re.compile(
    r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|DROP\s+TABLE|ALTER\s+TABLE|WHERE|JOIN)\b",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(
    r"(^#{1,6}\s+\S|\*\*[^*]+\*\*|__[^_]+__|\[[^\]]+\]\([^)]+\)|^\s*[-*+]\s+\S|^\s*\d+\.\s+\S)",
    re.MULTILINE,
)
_URL_RE = re.compile(r"\bhttps?://[^\s]+|\bwww\.[^\s]+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
_LATEX_RE = re.compile(r"\$[^$]+\$|\\\[.*?\\\]|\\begin\{[^}]+\}|\\frac|\\sum|\\int")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+")
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s\"']+|(?:/[\w.\-]+){2,}|\./[\w./\-]+|~/[\w./\-]+)"
)
_SHELL_CMD_RE = re.compile(
    r"^\s*[$#>]\s+\S|\b(sudo|cd|ls|grep|mkdir|rm\s+-rf|chmod|chown|curl|wget|git\s+\w+|pip\s+install|npm\s+install|apt-get|docker\s+\w+)\b"
)
_LIST_RE = re.compile(r"(^\s*[-*+]\s+\S|^\s*\d+[.)]\s+\S)", re.MULTILINE)
_NUMBERS_RE = re.compile(r"\d")


def _search(pattern: re.Pattern, text: str) -> bool:
    """Return True if the pattern matches anywhere in the text."""
    return bool(pattern.search(text))


def extract_structural(prompt: str) -> dict:
    """Extract structural features from a prompt using regex detection.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A dictionary of boolean features describing structural content
        present in the prompt (code, markup, URLs, tables, etc.).
    """
    text = prompt if isinstance(prompt, str) else ""

    return {
        "has_code": _search(_CODE_BLOCK_RE, text),
        "has_json": _search(_JSON_RE, text),
        "has_xml": _search(_XML_RE, text),
        "has_html": _search(_HTML_RE, text),
        "has_sql": _search(_SQL_RE, text),
        "has_markdown": _search(_MARKDOWN_RE, text),
        "has_url": _search(_URL_RE, text),
        "has_email": _search(_EMAIL_RE, text),
        "has_latex": _search(_LATEX_RE, text),
        "has_python_traceback": _search(_TRACEBACK_RE, text),
        "has_table": _search(_TABLE_RE, text),
        "has_file_path": _search(_FILE_PATH_RE, text),
        "has_shell_command": _search(_SHELL_CMD_RE, text),
        "has_list": _search(_LIST_RE, text),
        "has_numbers": _search(_NUMBERS_RE, text),
    }
