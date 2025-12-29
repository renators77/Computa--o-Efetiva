"""Local lexicon based sentiment/emotion analysis."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

LEXICON: Dict[str, Tuple[float, float, str]] = {
    "feliz": (0.9, 0.7, "alegria"),
    "felizmente": (0.8, 0.6, "alegria"),
    "fortunadamente": (0.7, 0.6, "alegria"),
    "infelizmente": (-0.7, 0.8, "tristeza"),
    "triste": (-0.8, 0.9, "tristeza"),
    "tristeza": (-0.8, 0.9, "tristeza"),
    "raiva": (-0.9, 0.8, "raiva"),
    "furioso": (-0.9, 0.8, "raiva"),
    "medo": (-0.6, 0.7, "medo"),
    "assustado": (-0.6, 0.7, "medo"),
    "surpresa": (0.1, 0.5, "surpresa"),
    "surpreso": (0.1, 0.5, "surpresa"),
    "adoro": (0.8, 0.8, "alegria"),
    "amo": (0.9, 0.8, "alegria"),
    "odio": (-0.9, 0.9, "raiva"),
    "odeio": (-0.9, 0.9, "raiva"),
    "great": (0.8, 0.6, "alegria"),
    "happy": (0.9, 0.7, "alegria"),
    "sad": (-0.8, 0.8, "tristeza"),
    "anger": (-0.9, 0.8, "raiva"),
    "scared": (-0.6, 0.7, "medo"),
    "afraid": (-0.6, 0.7, "medo"),
    "unfortunately": (-0.6, 0.7, "tristeza"),
    "fortunately": (0.6, 0.5, "alegria"),
    "hate": (-0.9, 0.9, "raiva"),
}

EMOTION_PRIORITY = [
    "alegria",
    "tristeza",
    "raiva",
    "medo",
    "surpresa",
]


class SentimentAnalyzer:
    """Aggregates lexicon scores into document-level metrics."""

    def analyze(self, tokens: list[str]) -> tuple[float, float, str]:
        if not tokens:
            return 0.0, 0.0, "neutro"
        total = 0.0
        subj_total = 0.0
        hits = 0
        emotions = Counter()
        for token in tokens:
            entry = LEXICON.get(token)
            if not entry:
                continue
            polarity, subjectivity, emotion = entry
            total += polarity
            subj_total += subjectivity
            hits += 1
            emotions[emotion] += 1
        if not hits:
            return 0.0, 0.1, "neutro"
        polarity_score = total / hits
        subjectivity_score = subj_total / hits
        emotion = self._select_emotion(emotions)
        return polarity_score, subjectivity_score, emotion

    @staticmethod
    def _select_emotion(counter: Counter) -> str:
        if not counter:
            return "neutro"
        # deterministic tie-breaking by priority list
        best_emotion = "neutro"
        best_count = 0
        for emotion in EMOTION_PRIORITY:
            count = counter.get(emotion, 0)
            if count > best_count:
                best_count = count
                best_emotion = emotion
        if best_count == 0:
            return "neutro"
        return best_emotion


__all__ = ["SentimentAnalyzer"]
