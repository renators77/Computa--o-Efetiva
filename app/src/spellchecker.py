"""Light-weight dictionary based spellchecker."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, List, Sequence, Set

_WORD_RE = re.compile(r"\b[\wáéíóúàâêôãõçñ]+\b", re.IGNORECASE)
_DEFAULT_DICTS = [
    Path("data/dictionaries/pt.txt"),
    Path("data/dictionaries/en.txt"),
]


@dataclass
class Correction:
    original: str
    corrected: str
    position: int


class SpellChecker:
    def __init__(
        self,
        dictionary_paths: Sequence[Path] | None = None,
        cutoff: float = 0.8,
    ) -> None:
        self.cutoff = cutoff
        self.words: Set[str] = set()
        paths = dictionary_paths or _DEFAULT_DICTS
        for path in paths:
            self._load_dictionary(path)

    def _load_dictionary(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                word = line.strip().lower()
                if word:
                    self.words.add(word)

    def _suggest(self, word: str) -> str | None:
        if not self.words:
            return None
        matches = get_close_matches(word.lower(), self.words, n=1, cutoff=self.cutoff)
        return matches[0] if matches else None

    @staticmethod
    def _preserve_case(original: str, suggestion: str) -> str:
        if original.isupper():
            return suggestion.upper()
        if original[:1].isupper():
            return suggestion.capitalize()
        return suggestion

    def correct_sentence(self, text: str) -> tuple[str, List[Correction]]:
        if not text:
            return "", []
        corrections: List[Correction] = []
        output: List[str] = []
        cursor = 0
        for match in _WORD_RE.finditer(text):
            start, end = match.span()
            word = match.group(0)
            output.append(text[cursor:start])
            cursor = end
            lower = word.lower()
            if lower in self.words:
                output.append(word)
                continue
            suggestion = self._suggest(lower)
            if not suggestion:
                output.append(word)
                continue
            corrected = self._preserve_case(word, suggestion)
            output.append(corrected)
            corrections.append(Correction(original=word, corrected=corrected, position=start))
        output.append(text[cursor:])
        return "".join(output), corrections


__all__ = ["SpellChecker", "Correction"]
