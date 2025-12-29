# Projeto — Análise de Frases (NLP Local)

## 1. Objetivo
Desenvolver um sistema **100% local** para análise de frases escritas (PT/EN), capaz de:
- Corrigir erros ortográficos simples;
- Interpretar frases e extrair parâmetros linguísticos;
- Identificar polaridade, subjetividade e emoção;
- Distinguir afirmações, negações e perguntas;
- Classificar frases como **pessoais** ou **factuais**;
- Produzir um output estruturado e explicável (JSON).

> Nota: Este módulo trata **exclusivamente input textual**. A análise de voz está fora do escopo atual.

---

## 2. Funcionalidades
Para cada frase de entrada, o sistema devolve:

1. **Frase corrigida** (se aplicável);
2. **Tipo de frase**:
   - Afirmação
   - Negação
   - Pergunta
   - Exclamação
3. **Natureza da frase**:
   - Pessoal
   - Factual
   - (Opcional) Mista
4. **Polaridade**:
   - Negativa ↔ Positiva (valor contínuo)
5. **Subjetividade**:
   - Objetiva ↔ Subjetiva (valor contínuo)
6. **Emoção textual**:
   - Alegria
   - Tristeza
   - Raiva
   - Medo
   - Surpresa
   - Neutro
7. **Evidências linguísticas** usadas na classificação.

---

## 3. Exemplos de funcionamento

| Frase | Resultado |
|------|----------|
| “A Nicole anda de bicicleta durante a tarde” | Afirmação · Factual · Neutro |
| “Não sei quando vou viajar infelizmente” | Negação · Pessoal · Tristeza |
| “Felizmente não chumbaram todos os alunos” | Negação · Factual · Alegria |

---

## 4. Arquitetura do Sistema

Frase → Normalização → Correção → NLP (parse) → Classificação (regras + modelo) → JSON output


---

## 5. Pipeline de Processamento

### 5.1 Normalização
- Remoção de espaços redundantes;
- Normalização de pontuação;
- Preservação da frase original.

### 5.2 Correção Ortográfica
- Correção baseada em palavras semelhantes (distância de edição);
- Suporte PT/EN;
- Registo explícito das correções efetuadas.

### 5.3 Análise Linguística
- Tokenização, lematização e POS-tagging;
- Análise de dependências sintáticas;
- Deteção de:
  - Negações;
  - Primeira pessoa;
  - Marcadores de opinião;
  - Marcadores factuais.

### 5.4 Classificação por Regras
- **Afirmação / Negação / Pergunta**;
- **Pessoal vs Factual**, com base em pronomes e verbos;
- Regras determinísticas e explicáveis.

### 5.5 Sentimento e Emoção
- Modelo local para polaridade (positivo / negativo / neutro);
- Deteção de emoções através de palavras-chave;
- Fusão de resultados modelo + regras.

---

## 6. Formato de Output

```json
{
  "original": "Não sei quando vou viajar infelizmente",
  "normalizada": "não sei quando vou viajar infelizmente",
  "corrigida": "Não sei quando vou viajar, infelizmente.",
  "correcoes": [
    { "from": "infelizmente", "to": "infelizmente", "pos": 26 }
  ],
  "tipo": "negação",
  "pessoal_factual": "pessoal",
  "polaridade": -0.72,
  "subjetividade": 0.81,
  "emocao": "tristeza",
  "evidencias": ["não sei", "1ª pessoa", "infelizmente"],
  "debug_features": {
    "has_negation": true,
    "first_person": true
  }
}
```
## 7 estrutura 
src/
 ├─ normalizer.py
 ├─ spellchecker.py
 ├─ nlp_parser.py
 ├─ rules.py
 ├─ sentiment.py
 ├─ pipeline.py
tests/
 ├─ test_examples.py
data/
 ├─ dictionaries/
 └─ test_sentences.json
README.md
