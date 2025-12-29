"""
Módulo de processamento de voz usando Whisper (Texto) e Librosa (Características de Voz).
Configurado para modelo 'base' e força apenas PT ou EN.
"""
import whisper
import os
import librosa
import numpy as np
import streamlit as st

class AudioTranscriber:
    def __init__(self, model_size="base"):
        print(f"A carregar modelo Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcreve áudio. Se detetar Inglês, mantém. 
        Se detetar qualquer outra coisa, força Português.
        """
        if not os.path.exists(audio_path):
            return {"text": "", "language": "unknown"}
        
        # 1. Primeira tentativa: Deteção Automática
        result = self.model.transcribe(audio_path, fp16=False)
        detected_lang = result["language"]
        text = result["text"].strip()

        # 2. Lógica de Restrição (Só PT ou EN)
        if detected_lang == 'en':
            return {"text": text, "language": "en"}
        elif detected_lang == 'pt':
            return {"text": text, "language": "pt"}
        else:
            print(f"Idioma '{detected_lang}' detetado. A forçar PT...")
            result_forced = self.model.transcribe(audio_path, fp16=False, language="pt")
            return {
                "text": result_forced["text"].strip(),
                "language": "pt (forçado)"
            }

    def analyze_voice_features(self, audio_path: str) -> dict:
        """
        Analisa características físicas da voz (Pitch e Energia) usando Librosa.
        """
        if not os.path.exists(audio_path):
            return {"tom": "Desconhecido", "intensidade": "N/A"}

        try:
            # Carregar áudio (apenas primeiros 5s)
            y, sr = librosa.load(audio_path, duration=5) 
        except Exception:
            return {"emoção_voz": "Erro", "detalhes": "Ficheiro inválido", "energia": 0, "pitch": 0}
        
        # 1. Energia (Volume - RMS)
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)
        
        # 2. Pitch (Frequência)
        # piptrack é mais robusto que zero-crossing
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Filtrar apenas as frequências com magnitude forte (ignorar ruído)
        indices = magnitudes > np.median(magnitudes)
        avg_pitch = np.mean(pitches[indices]) if np.any(indices) else 0

        # --- Lógica de Decisão (Calibrada) ---
        voice_emotion = "Neutro"
        explanation = "Tom de voz equilibrado."
        
        # Calibração de limites
        # Energia: < 0.015 é muito baixo (sussurro/tristeza), > 0.05 é alto (grito/alegria)
        limit_energy_low = 0.015
        limit_energy_high = 0.04
        
        # Pitch: Homens ~100-150Hz, Mulheres ~200-250Hz.
        # Acima de 280-300Hz começa a ser agudo/exaltado para ambos.
        limit_pitch_low = 140 
        limit_pitch_high = 280 

        if avg_energy > limit_energy_high:
            if avg_pitch > limit_pitch_high:
                voice_emotion = "Exaltado / Alegria / Pânico"
                explanation = "Volume alto e tom muito agudo."
            else:
                voice_emotion = "Raiva / Assertivo"
                explanation = "Volume alto e tom firme."
        
        elif avg_energy < limit_energy_low:
            if avg_pitch < limit_pitch_low:
                voice_emotion = "Tristeza / Cansaço / Tédio"
                explanation = "Volume baixo e tom grave."
            else:
                voice_emotion = "Calmo / Tímido"
                explanation = "Volume baixo."
        
        else:
            # Energia média (Conversa normal)
            voice_emotion = "Neutro / Conversa Normal"
            explanation = "Tom e volume moderados."

        return {
            "emoção_voz": voice_emotion,
            "detalhes": explanation,
            "energia": round(float(avg_energy), 4),
            "pitch": round(float(avg_pitch), 2)
        }

@st.cache_resource
def get_transcriber():
    return AudioTranscriber(model_size="base")