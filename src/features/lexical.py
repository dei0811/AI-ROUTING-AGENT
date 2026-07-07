"""Lexical feature extraction module.

This module computes surface-level lexical statistics from a raw prompt
string (character counts, word counts, punctuation usage, etc.). It has
no dependencies on other feature modules and can be tested in isolation.
"""

import re
import string


def extract_lexical(prompt: str) -> dict:
    """Extract lexical features from a prompt.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A dictionary of lexical features. Keys are feature names and
        values are numbers (int/float) representing lexical statistics
        of the given prompt.
    """
    text = prompt if isinstance(prompt, str) else ""

    num_chars = len(text)

    words = text.split()
    num_words = len(words)

    # Approximate tokenization using whitespace splitting.
    num_tokens = num_words

    word_lengths = [len(w.strip(string.punctuation)) for w in words]
    word_lengths = [length for length in word_lengths if length > 0]

    avg_word_length = (
        sum(word_lengths) / len(word_lengths) if word_lengths else 0.0
    )
    max_word_length = max(word_lengths) if word_lengths else 0

    # Sentences are approximated by splitting on ., !, ? followed by
    # whitespace or end of string.
    sentence_candidates = re.split(r"[.!?]+", text)
    num_sentences = len([s for s in sentence_candidates if s.strip()])

    lines = text.splitlines()
    num_lines = len(lines)

    # Paragraphs are separated by one or more blank lines.
    paragraphs = re.split(r"\n\s*\n", text)
    num_paragraphs = len([p for p in paragraphs if p.strip()])

    punctuation_count = sum(1 for ch in text if ch in string.punctuation)
    question_marks = text.count("?")
    exclamation_marks = text.count("!")
    digit_count = sum(1 for ch in text if ch.isdigit())

    alpha_chars = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = (
        sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )

    normalized_words = [w.strip(string.punctuation).lower() for w in words]
    normalized_words = [w for w in normalized_words if w]
    unique_word_ratio = (
        len(set(normalized_words)) / len(normalized_words)
        if normalized_words
        else 0.0
    )

    return {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_tokens": num_tokens,
        "avg_word_length": round(avg_word_length, 4),
        "max_word_length": max_word_length,
        "num_sentences": num_sentences,
        "num_lines": num_lines,
        "num_paragraphs": num_paragraphs,
        "punctuation_count": punctuation_count,
        "question_marks": question_marks,
        "exclamation_marks": exclamation_marks,
        "digit_count": digit_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
    }

