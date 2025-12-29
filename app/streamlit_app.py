"""Minimal Streamlit UI for the local NLP pipeline."""
from __future__ import annotations

import streamlit as st
import os

from src.pipeline import NLPPipeline
from src.audio import get_transcriber 

st.set_page_config(page_title="Analise de Frases", layout="wide")
st.title("Analisador local de frases e voz")
st.caption("Processamento: Whisper (Texto) + Librosa (Tom de Voz) + Regras NLP")

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = NLPPipeline()

pipeline = st.session_state.pipeline

# --- SECÇÃO DE ÁUDIO ---
st.subheader("🎤 Entrada de Voz")
st.info("Usa o gravador abaixo. Clica na bola vermelha para gravar e no quadrado para parar.")

audio_value = st.audio_input("Grave a sua voz aqui")

transcribed_text = ""
voice_analysis = None

if audio_value:
    audio_file_path = "gravacao_temp.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_value.read())
    
    with st.spinner("A processar áudio (Whisper + Librosa)..."):
        try:
            transcriber = get_transcriber()
            
            # 1. Transcrever Texto (Com filtro PT/EN)
            result_whisper = transcriber.transcribe(audio_file_path)
            
            text_from_voice = result_whisper["text"]
            detected_lang = result_whisper["language"]
            
            transcribed_text = text_from_voice
            
            if "pt" in detected_lang:
                st.success(f"Idioma detetado: Português ({detected_lang})")
            elif "en" in detected_lang:
                st.info(f"Idioma detetado: Inglês ({detected_lang})")
            else:
                st.warning(f"Idioma: {detected_lang}")

            # 2. Analisar Emoção na Voz
            voice_analysis = transcriber.analyze_voice_features(audio_file_path)
            
        except Exception as e:
            st.error(f"Erro: {e}. (Verifica se tens o FFmpeg instalado!)")

# --- MOSTRAR RESULTADOS DA VOZ ---
if voice_analysis:
    with st.expander("📊 Análise Acústica da Voz (Tom e Intensidade)", expanded=True):
        k1, k2, k3 = st.columns(3)
        k1.metric("Emoção Detetada na Voz", voice_analysis["emoção_voz"])
        k2.metric("Intensidade (Volume)", voice_analysis["energia"])
        k3.metric("Frequência Média (Hz)", voice_analysis["pitch"])
        st.caption(f"Explicação: {voice_analysis['detalhes']}")

# --- SECÇÃO DE TEXTO ---
st.divider()
st.subheader("📝 Análise Linguística (Texto)")

default_text = transcribed_text if transcribed_text else "A Nicole anda de bicicleta durante a tarde"
text = st.text_area("Texto a analisar", value=default_text, height=100)

if st.button("Analisar Texto") or (text.strip() and transcribed_text):
    resultado = pipeline.process(text)
    
    # Criar colunas para visualização
    c1, c2, c3 = st.columns(3)
    c1.metric("Tipo de Frase", resultado["tipo"].title())
    c2.metric("Natureza", resultado["pessoal_factual"].title())
    
    # --- Mapeamento Visual de Emoções ---
    emocao_txt = resultado["emocao"].lower()
    
    # Definir Ícone e Cor
    emoji = "😐"
    delta_color = "off" # Cinza por defeito
    
    if emocao_txt == "alegria":
        emoji = "😄"
        delta_color = "normal" # Verde
    elif emocao_txt == "tristeza":
        emoji = "😢"
        delta_color = "inverse" # Vermelho
    elif emocao_txt == "raiva":
        emoji = "😡"
        delta_color = "inverse"
    elif emocao_txt == "medo":
        emoji = "😨"
        delta_color = "inverse"
    elif emocao_txt == "surpresa":
        emoji = "😲"
        delta_color = "off"
    elif emocao_txt == "nojo":
        emoji = "🤢"
        delta_color = "inverse"
    elif emocao_txt == "confiança":
        emoji = "🤝"
        delta_color = "normal"
    elif emocao_txt == "antecipação":
        emoji = "🤔"
        delta_color = "off"

    # Mostrar métrica com Emoji
    c3.metric("Sentimento (Texto)", f"{emoji} {emocao_txt.title()}", delta_color=delta_color)

    st.write("---")
    col_A, col_B = st.columns(2)
    
    with col_A:
        st.write("**Frase Corrigida:**")
        if resultado["corrigida"] != resultado["original"]:
            st.warning(resultado["corrigida"])
        else:
            st.info(resultado["corrigida"])

    with col_B:
        st.write("**Parâmetros:**")
        st.write(f"Polaridade: `{resultado['polaridade']}` (-1 a 1)")
        st.write(f"Subjetividade: `{resultado['subjetividade']}` (0 a 1)")

    with st.expander("Ver JSON Técnico"):
        st.json(resultado)