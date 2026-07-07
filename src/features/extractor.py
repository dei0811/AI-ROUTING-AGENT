"""Feature extraction orchestrator.

Combines the outputs of all independent feature modules (lexical,
language, structural, task, complexity) into a single flat dictionary
of features describing a prompt. Intended to be consumed by the
XGBoost-based routing model.
"""

from .complexity import extract_complexity
from .language import extract_language
from .lexical import extract_lexical
from .structural import extract_structural
from .task import extract_task


def extract_features(prompt: str) -> dict:
    """Extract the full feature set for a given prompt.

    Args:
        prompt: Raw user prompt string.

    Returns:
        A single dictionary containing all features produced by the
        lexical, language, structural, task, and complexity extractors.
    """
    features = {}

    features.update(extract_lexical(prompt))
    features.update(extract_language(prompt))
    features.update(extract_structural(prompt))
    features.update(extract_task(prompt))
    features.update(extract_complexity(prompt))

    return features
