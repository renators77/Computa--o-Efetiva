"""
Spellchecker robusto usando a biblioteca 'pyspellchecker'.
Suporta PT e EN simultaneamente.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set
from spellchecker import SpellChecker as PySpellChecker

_WORD_RE = re.compile(r"\b[\wáéíóúàâêôãõçñ']+\b", re.IGNORECASE)

@dataclass
class Correction:
    original: str
    corrected: str
    position: int

class SpellChecker:
    def __init__(self) -> None:
        # Carregar dicionários completos
        print("A carregar dicionários de correção (PT/EN)...")
        self.spell_pt = PySpellChecker(language='pt')
        self.spell_en = PySpellChecker(language='en')
        
        # Adicionar termos técnicos ou específicos que o dicionário possa não ter
        custom_words = {"streamlit", "app", "python", "code", "olá", "whisper", "software"}
        self.spell_pt.word_frequency.load_words(custom_words)
        self.spell_en.word_frequency.load_words(custom_words)

    def _is_known(self, word: str) -> bool:
        """Verifica se a palavra existe em PT ou EN."""
        return (word.lower() in self.spell_pt) or (word.lower() in self.spell_en)

    def _suggest(self, word: str) -> str:
        """
        Tenta corrigir. Prioridade:
        1. Se a palavra for muito parecida com uma PT, corrige para PT.
        2. Se for muito parecida com uma EN, corrige para EN.
        """
        # Tenta correção em PT primeiro (regra do projeto: default PT)
        res_pt = self.spell_pt.correction(word)
        
        # Se o PT não mudou nada ou devolveu a mesma, confiamos
        if res_pt == word:
            return word
            
        # Se o PT mudou, vamos ver se em Inglês a palavra original existia
        # (ex: "date" em ingles não deve virar "data" em pt se o contexto for misto)
        if word.lower() in self.spell_en:
            return word
            
        return res_pt if res_pt else word

    @staticmethod
    def _preserve_case(original: str, suggestion: str) -> str:
        if original.isupper():
            return suggestion.upper()
        if original[0].isupper():
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
            
            # Adicionar o texto entre palavras (espaços, pontuação)
            output.append(text[cursor:start])
            cursor = end
            
            # Verificar se a palavra existe
            if self._is_known(word):
                output.append(word)
                continue
                
            # Se não existe, tentar corrigir
            suggestion = self._suggest(word)
            
            # Se não houver sugestão ou for igual, mantém
            if not suggestion or suggestion.lower() == word.lower():
                output.append(word)
                continue
                
            # Aplicar correção mantendo maiúsculas/minúsculas
            corrected = self._preserve_case(word, suggestion)
            output.append(corrected)
            corrections.append(Correction(original=word, corrected=corrected, position=start))
            
        output.append(text[cursor:])
        return "".join(output), corrections