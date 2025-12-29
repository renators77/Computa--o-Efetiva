"""Rule based reasoning for type/nature/evidence extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .nlp_parser import ParsedSentence


@dataclass
class ClassificationResult:
    sentence_type: str
    nature: str
    evidences: List[str]


class RuleBasedClassifier:
    def classify(self, parsed: ParsedSentence, text: str) -> ClassificationResult:
        evidences: List[str] = []
        if parsed.negation_terms:
            evidences.extend(parsed.negation_terms)
        if parsed.question_terms:
            evidences.extend(parsed.question_terms)
        if parsed.opinion_markers:
            evidences.extend(parsed.opinion_markers)
        if parsed.first_person:
            evidences.append("1ª pessoa")
        if parsed.is_question:
            evidences.append("pontuação ?")
        if parsed.is_exclamation:
            evidences.append("pontuação !")

        sentence_type = self._sentence_type(parsed)
        nature = self._sentence_nature(parsed)
        return ClassificationResult(
            sentence_type=sentence_type,
            nature=nature,
            evidences=list(dict.fromkeys(evidences)),  # keep order, deduplicate
        )

    @staticmethod
    def _sentence_type(parsed: ParsedSentence) -> str:
        if parsed.is_question:
            return "pergunta"
        if parsed.has_negation:
            return "negação"
        if parsed.is_exclamation:
            return "exclamação"
        return "afirmação"

    @staticmethod
    def _sentence_nature(parsed: ParsedSentence) -> str:
        if parsed.first_person and parsed.factual_markers:
            return "mista"
        if parsed.first_person or parsed.opinion_markers:
            return "pessoal"
        return "factual"


__all__ = ["RuleBasedClassifier", "ClassificationResult"]
