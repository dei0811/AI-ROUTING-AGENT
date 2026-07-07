"""Language feature extraction module.

Detects language-related characteristics of a prompt: detected language
code, detection confidence, lexical diversity, and average sentence
length. Uses ``langdetect`` when available; falls back to ``None`` for
any feature that cannot be computed.
"""

import re
import string

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def _detect_language(text: str):
    """Detect language and confidence using langdetect if possible.

    Args:
        text: Input text to analyze.

    Returns:
        A tuple (language, confidence). Both are ``None`` if detection
        is not possible (empty text, missing dependency, or error).
    """
    if not _LANGDETECT_AVAILABLE or not text or not text.strip():
        return None, None

    try:
        results = detect_langs(text)
        if not results:
            return None, None
        best = results[0]
        return best.lang, round(float(best.prob), 4)
    except Exception:
        return None, None


def _lexical_diversity(text: str):
    """Compute the ratio of unique words to total words.

    Returns:
        A float in [0, 1], or ``None`` if the text has no words.
    """
    words = [
        w.strip(string.punctuation).lower()
        for w in text.split()
        if w.strip(string.punctuation)
    ]
    if not words:
        return None
    return round(len(set(words)) / len(words), 4)


def _average_sentence_length(text: str):
    """Compute the average number of words per sentence.

    Returns:
        A float, or ``None`` if no sentences/words are found.
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return None

    total_words = sum(len(s.split()) for s in sentences)
    if total_words == 0:
        return None

    return round(total_words / len(sentences), 4)


def extract_language(prompt: str) -> dict:
    """Extract language-related features from a prompt.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A dictionary with keys: ``language``, ``language_confidence``,
        ``lexical_diversity``, ``average_sentence_length``. Any feature
        that cannot be computed is set to ``None``.
    """
    text = prompt if isinstance(prompt, str) else ""

    language, confidence = _detect_language(text)

    return {
        "language": language,
        "language_confidence": confidence,
        "lexical_diversity": _lexical_diversity(text),
        "average_sentence_length": _average_sentence_length(text),
    }
