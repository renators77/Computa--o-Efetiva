"""Text normalization utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
DOUBLE_PUNCT_RE = re.compile(r"([.!?]){2,}")
MULTISPACE_PUNCT_RE = re.compile(r"([,.;:!?])(?!\s)")


def _cleanup_spacing(text: str) -> str:
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = MULTISPACE_PUNCT_RE.sub(r"\1 ", text)
    return text


@dataclass
class NormalizedText:
    original: str
    cleaned: str
    normalized: str
    steps: List[str]


class Normalizer:
    """Performs deterministic cleanup before parsing."""

    def normalize(self, text: str) -> NormalizedText:
        if text is None:
            text = ""
        steps: List[str] = []
        original = text
        cleaned = text.strip()
        if cleaned != text:
            steps.append("trim_whitespace")
        cleaned = WHITESPACE_RE.sub(" ", cleaned)
        cleaned = _cleanup_spacing(cleaned)
        cleaned = DOUBLE_PUNCT_RE.sub(lambda m: m.group(1), cleaned)
        normalized = cleaned.lower()
        steps.append("collapse_spaces")
        steps.append("normalize_case")
        return NormalizedText(
            original=original,
            cleaned=cleaned,
            normalized=normalized,
            steps=steps,
        )


__all__ = ["Normalizer", "NormalizedText"]
