"""Minimal Streamlit UI for the local NLP pipeline."""
from __future__ import annotations

import streamlit as st

from src.pipeline import NLPPipeline

st.set_page_config(page_title="Analise de Frases", layout="wide")
st.title("Analisador local de frases")
st.caption("Processamento totalmente local utilizando regras e léxicos simples.")

pipeline = NLPPipeline()
text = st.text_area("Introduza a frase", "Não sei quando vou viajar infelizmente")

if st.button("Analisar") or text.strip():
    resultado = pipeline.process(text)
    st.subheader("Resultado")
    st.json(resultado)

    st.subheader("Resumo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tipo", resultado["tipo"].title())
    col2.metric("Natureza", resultado["pessoal_factual"].title())
    col3.metric("Emoção", resultado["emocao"].title())

    st.subheader("Evidências")
    st.write(", ".join(resultado["evidencias"] or ["Sem evidências explícitas"]))
