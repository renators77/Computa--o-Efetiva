"""Local lexicon based sentiment/emotion analysis with negation handling."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

# Importar as negações do parser
from .nlp_parser import NEGATIONS

# Estrutura: "palavra": (Polaridade, Subjetividade, Emoção)
# Polaridade: -1.0 (Negativo) a 1.0 (Positivo)
LEXICON: Dict[str, Tuple[float, float, str]] = {
    # --- ALEGRIA (JOY) ---
    "feliz": (0.9, 0.7, "alegria"), "contente": (0.8, 0.6, "alegria"),
    "excelente": (0.9, 0.6, "alegria"), "bom": (0.6, 0.5, "alegria"),
    "adoro": (0.9, 0.8, "alegria"), "amo": (1.0, 0.9, "alegria"),
    "fantástico": (0.9, 0.8, "alegria"), "maravilhoso": (1.0, 0.9, "alegria"),
    "happy": (0.9, 0.7, "alegria"), "great": (0.8, 0.6, "alegria"),
    "love": (0.9, 0.8, "alegria"), "good": (0.6, 0.5, "alegria"),
    "amazing": (0.9, 0.8, "alegria"), "fun": (0.8, 0.6, "alegria"),
    "ganhei": (0.9, 0.5, "alegria"), "win": (0.9, 0.5, "alegria"),
    "parabéns": (0.9, 0.6, "alegria"), "congrats": (0.9, 0.6, "alegria"),
    "top": (0.7, 0.6, "alegria"), "fixe": (0.7, 0.6, "alegria"),
    "legal": (0.6, 0.5, "alegria"), "espetáculo": (0.9, 0.8, "alegria"),
    "brutal": (0.8, 0.7, "alegria"), "lindo": (0.8, 0.7, "alegria"),
    "obrigado": (0.5, 0.2, "alegria"), "thanks": (0.5, 0.2, "alegria"),
    "excited": (0.8, 0.7, "alegria"), "entusiasmado": (0.8, 0.7, "alegria"),
    "orgulho": (0.9, 0.7, "alegria"), "proud": (0.9, 0.7, "alegria"),
    "rir": (0.7, 0.5, "alegria"), "laugh": (0.7, 0.5, "alegria"),
    "lol": (0.6, 0.4, "alegria"), "haha": (0.6, 0.4, "alegria"),

    # --- TRISTEZA (SADNESS) ---
    "infelizmente": (-0.7, 0.8, "tristeza"), "triste": (-0.9, 0.9, "tristeza"),
    "magoado": (-0.8, 0.8, "tristeza"), "deprimido": (-0.9, 0.9, "tristeza"),
    "pena": (-0.5, 0.6, "tristeza"), "chumbar": (-0.8, 0.5, "tristeza"),
    "sad": (-0.8, 0.8, "tristeza"), "bad": (-0.7, 0.6, "tristeza"),
    "sorry": (-0.5, 0.5, "tristeza"), "miss": (-0.6, 0.7, "tristeza"),
    "cry": (-0.8, 0.8, "tristeza"), "chorar": (-0.8, 0.8, "tristeza"),
    "alone": (-0.7, 0.8, "tristeza"), "sozinho": (-0.7, 0.8, "tristeza"),
    "perdi": (-0.8, 0.6, "tristeza"), "lost": (-0.8, 0.6, "tristeza"),
    "saudade": (-0.6, 0.8, "tristeza"), "luto": (-1.0, 0.9, "tristeza"),
    "desiludido": (-0.7, 0.7, "tristeza"), "disappointed": (-0.7, 0.7, "tristeza"),
    "cansei": (-0.6, 0.6, "tristeza"), "tired": (-0.5, 0.5, "tristeza"),
    "pobre": (-0.4, 0.2, "tristeza"), "poor": (-0.4, 0.2, "tristeza"),

    # --- RAIVA (ANGER) ---
    "raiva": (-0.9, 0.8, "raiva"), "furioso": (-0.9, 0.8, "raiva"),
    "irritado": (-0.7, 0.8, "raiva"), "chato": (-0.6, 0.7, "raiva"),
    "odeio": (-1.0, 0.9, "raiva"), "detesto": (-0.9, 0.9, "raiva"),
    "estúpido": (-0.8, 0.9, "raiva"), "horrível": (-0.9, 0.8, "raiva"),
    "anger": (-0.9, 0.8, "raiva"), "hate": (-0.9, 0.9, "raiva"),
    "annoying": (-0.7, 0.7, "raiva"), "terrible": (-0.9, 0.8, "raiva"),
    "mad": (-0.8, 0.8, "raiva"), "lento": (-0.5, 0.4, "raiva"),
    "burro": (-0.8, 0.9, "raiva"), "idiot": (-0.8, 0.9, "raiva"),
    "merda": (-0.9, 0.9, "raiva"), "shit": (-0.9, 0.9, "raiva"),
    "porra": (-0.8, 0.9, "raiva"), "fuck": (-0.9, 0.9, "raiva"),
    "injusto": (-0.7, 0.6, "raiva"), "unfair": (-0.7, 0.6, "raiva"),
    "farto": (-0.7, 0.7, "raiva"), "fed up": (-0.7, 0.7, "raiva"),

    # --- MEDO (FEAR) ---
    "medo": (-0.8, 0.8, "medo"), "assustado": (-0.7, 0.8, "medo"),
    "perigoso": (-0.7, 0.4, "medo"), "nervoso": (-0.5, 0.7, "medo"),
    "ansioso": (-0.5, 0.8, "medo"), "pânico": (-0.9, 0.9, "medo"),
    "scared": (-0.8, 0.8, "medo"), "afraid": (-0.7, 0.8, "medo"),
    "fear": (-0.8, 0.8, "medo"), "danger": (-0.8, 0.5, "medo"),
    "horror": (-0.9, 0.9, "medo"), "socorro": (-0.5, 0.6, "medo"),
    "help": (-0.4, 0.6, "medo"), "correr": (-0.2, 0.4, "medo"),
    "fugir": (-0.5, 0.6, "medo"), "run": (-0.2, 0.4, "medo"),
    "stress": (-0.6, 0.5, "medo"), "tenso": (-0.5, 0.5, "medo"),

    # --- SURPRESA (SURPRISE) ---
    "surpresa": (0.3, 0.6, "surpresa"), "espantado": (0.4, 0.7, "surpresa"),
    "choque": (-0.2, 0.8, "surpresa"), "incrivel": (0.8, 0.7, "surpresa"),
    "wow": (0.5, 0.6, "surpresa"), "surprise": (0.3, 0.6, "surpresa"),
    "shocked": (-0.2, 0.8, "surpresa"), "really?": (0.1, 0.5, "surpresa"),
    "sério?": (0.1, 0.5, "surpresa"), "nossa": (0.3, 0.6, "surpresa"),
    "eita": (0.1, 0.5, "surpresa"), "omg": (0.2, 0.6, "surpresa"),
    "impossível": (-0.1, 0.4, "surpresa"), "impossible": (-0.1, 0.4, "surpresa"),

    # --- NOJO (DISGUST) ---
    "nojo": (-0.9, 0.9, "nojo"), "nojento": (-0.9, 0.9, "nojo"),
    "podre": (-0.9, 0.6, "nojo"), "lixo": (-0.8, 0.8, "nojo"),
    "enjoo": (-0.6, 0.8, "nojo"), "repugnante": (-0.9, 0.8, "nojo"),
    "disgust": (-0.9, 0.9, "nojo"), "gross": (-0.8, 0.8, "nojo"),
    "trash": (-0.8, 0.8, "nojo"), "sick": (-0.7, 0.8, "nojo"),
    "ew": (-0.6, 0.9, "nojo"), "nasty": (-0.8, 0.8, "nojo"),
    "vomitar": (-0.9, 0.9, "nojo"), "cheiro": (-0.3, 0.4, "nojo"), # contexto negativo usual

    # --- CONFIANÇA (TRUST) ---
    "confio": (0.8, 0.7, "confiança"), "verdade": (0.6, 0.2, "confiança"),
    "certo": (0.5, 0.3, "confiança"), "seguro": (0.7, 0.4, "confiança"),
    "amigo": (0.8, 0.6, "confiança"), "concordo": (0.6, 0.4, "confiança"),
    "trust": (0.8, 0.7, "confiança"), "true": (0.6, 0.2, "confiança"),
    "sure": (0.6, 0.3, "confiança"), "safe": (0.7, 0.4, "confiança"),
    "agree": (0.6, 0.4, "confiança"), "friend": (0.8, 0.6, "confiança"),
    "claro": (0.4, 0.2, "confiança"), "definitely": (0.5, 0.3, "confiança"),
    "líder": (0.5, 0.3, "confiança"), "apoio": (0.7, 0.5, "confiança"),

    # --- ANTECIPAÇÃO (ANTICIPATION) ---
    "espero": (0.4, 0.6, "antecipação"), "breve": (0.2, 0.2, "antecipação"),
    "ansiosamente": (0.6, 0.8, "antecipação"), "preparado": (0.5, 0.4, "antecipação"),
    "hope": (0.6, 0.6, "antecipação"), "wait": (0.0, 0.5, "antecipação"),
    "soon": (0.3, 0.2, "antecipação"), "ready": (0.5, 0.4, "antecipação"),
    "plan": (0.2, 0.1, "antecipação"), "plano": (0.2, 0.1, "antecipação"),
    "amanhã": (0.1, 0.1, "antecipação"), "tomorrow": (0.1, 0.1, "antecipação"),
    "vamos": (0.3, 0.3, "antecipação"), "let's": (0.3, 0.3, "antecipação"),
    "futuro": (0.4, 0.2, "antecipação"), "future": (0.4, 0.2, "antecipação"),
}

EMOTION_PRIORITY = ["raiva", "nojo", "medo", "tristeza", "alegria", "surpresa", "antecipação", "confiança"]

class SentimentAnalyzer:
    """Aggregates lexicon scores with negation handling."""

    def analyze(self, tokens: list[str]) -> tuple[float, float, str]:
        if not tokens:
            return 0.0, 0.0, "neutro"
            
        total_polarity = 0.0
        total_subjectivity = 0.0
        hits = 0
        emotions = Counter()
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            entry = LEXICON.get(token_lower)
            if not entry:
                continue
                
            polarity, subjectivity, emotion = entry
            
            # --- LÓGICA DE NEGAÇÃO ---
            if i > 0 and tokens[i-1].lower() in NEGATIONS:
                polarity = polarity * -1.0
                if emotion == "alegria": emotion = "tristeza"
                elif emotion == "tristeza": emotion = "alegria"
                elif emotion == "raiva": emotion = "calmo"
                elif emotion == "medo": emotion = "confiança"
                elif emotion == "confiança": emotion = "medo"
                elif emotion == "nojo": emotion = "neutro"
                elif emotion == "antecipação": emotion = "surpresa"
            # -------------------------

            total_polarity += polarity
            total_subjectivity += subjectivity
            hits += 1
            emotions[emotion] += 1

        if not hits:
            return 0.0, 0.1, "neutro"

        polarity_score = total_polarity / hits
        subjectivity_score = total_subjectivity / hits
        
        final_emotion = self._select_emotion(emotions)
        
        return polarity_score, subjectivity_score, final_emotion

    @staticmethod
    def _select_emotion(counter: Counter) -> str:
        if not counter:
            return "neutro"
        best_emotion = "neutro"
        best_count = 0
        for emotion in EMOTION_PRIORITY:
            count = counter.get(emotion, 0)
            if count > best_count:
                best_count = count
                best_emotion = emotion
        if best_count > 0:
            return best_emotion
        return counter.most_common(1)[0][0]

__all__ = ["SentimentAnalyzer"]