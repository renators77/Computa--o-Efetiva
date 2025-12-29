"""Simple local NLP parser with lightweight heuristics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

TOKEN_RE = re.compile(r"[\wáéíóúàâêôãõçñ']+", re.IGNORECASE)
NEGATIONS = {
    "não",
    "nao",
    "nunca",
    "jamais",
    "sem",
    "not",
    "never",
    "no",
}
QUESTION_TERMS = {
    "quem",
    "onde",
    "quando",
    "como",
    "porquê",
    "porque",
    "qual",
    "quanto",
    "why",
    "who",
    "where",
    "when",
    "how",
    "what",
}
FIRST_PERSON = {
    "eu",
    "meu",
    "minha",
    "acho",
    "penso",
    "sinto",
    "estou",
    "i",
    "me",
    "my",
    "mine",
    "feel",
    "think",
}
OPINION_MARKERS = {
    "acho",
    "penso",
    "sinto",
    "feel",
    "believe",
    "seems",
    "parece",
}
FACTUAL_MARKERS = {
    "dados",
    "pesquisa",
    "segundo",
    "relatório",
    "report",
    "diz",
    "informou",
    "mediu",
    "percent",
}


@dataclass
class ParsedSentence:
    tokens: List[str]
    lemmas: List[str]
    has_negation: bool
    is_question: bool
    is_exclamation: bool
    first_person: bool
    opinion_markers: List[str]
    factual_markers: List[str]
    negation_terms: List[str]
    question_terms: List[str]


class SimpleNLPParser:
    """Extracts shallow syntactic/semantic hints without external models."""

    def parse(self, text: str) -> ParsedSentence:
        tokens = TOKEN_RE.findall(text.lower())
        negation_terms = [t for t in tokens if t in NEGATIONS]
        question_terms = [t for t in tokens if t in QUESTION_TERMS]
        opinion_terms = [t for t in tokens if t in OPINION_MARKERS]
        factual_terms = [t for t in tokens if t in FACTUAL_MARKERS]
        has_negation = bool(negation_terms)
        is_question = text.strip().endswith("?") or bool(question_terms)
        is_exclamation = text.strip().endswith("!")
        first_person = any(t in FIRST_PERSON for t in tokens)
        lemmas = tokens  # placeholder without heavy dep
        return ParsedSentence(
            tokens=tokens,
            lemmas=lemmas,
            has_negation=has_negation,
            is_question=is_question,
            is_exclamation=is_exclamation,
            first_person=first_person,
            opinion_markers=opinion_terms,
            factual_markers=factual_terms,
            negation_terms=negation_terms,
            question_terms=question_terms,
        )


__all__ = ["SimpleNLPParser", "ParsedSentence"]
