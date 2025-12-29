"""NLP pipeline orchestrator."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .nlp_parser import SimpleNLPParser
from .normalizer import Normalizer
from .rules import RuleBasedClassifier
from .sentiment import SentimentAnalyzer
from .spellchecker import SpellChecker


class NLPPipeline:
    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.spellchecker = SpellChecker()
        self.parser = SimpleNLPParser()
        self.classifier = RuleBasedClassifier()
        self.sentiment = SentimentAnalyzer()

    def process(self, text: str) -> Dict[str, Any]:
        normalized = self.normalizer.normalize(text)
        corrected_text, corrections = self.spellchecker.correct_sentence(normalized.cleaned)
        parsed = self.parser.parse(corrected_text)
        classification = self.classifier.classify(parsed, corrected_text)
        polarity, subjectivity, emotion = self.sentiment.analyze(parsed.tokens)

        return {
            "original": normalized.original,
            "normalizada": normalized.normalized,
            "corrigida": corrected_text,
            "correcoes": [
                {"from": c.original, "to": c.corrected, "pos": c.position}
                for c in corrections
            ],
            "tipo": classification.sentence_type,
            "pessoal_factual": classification.nature,
            "polaridade": round(polarity, 2),
            "subjetividade": round(subjectivity, 2),
            "emocao": emotion,
            "evidencias": classification.evidences,
            "debug_features": {
                "has_negation": parsed.has_negation,
                "is_question": parsed.is_question,
                "is_exclamation": parsed.is_exclamation,
                "first_person": parsed.first_person,
            },
        }


def analyze_sentence(sentence: str) -> Dict[str, Any]:
    """Convenience helper for quick scripts/tests."""
    pipeline = NLPPipeline()
    return pipeline.process(sentence)


__all__ = ["NLPPipeline", "analyze_sentence"]
