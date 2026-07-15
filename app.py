"""Playground de geração de letras de música brasileira.

Carrega o GPT-2 em português fine-tunado com letras de música e permite
experimentar os parâmetros de geração (temperatura, top-p, top-k etc.).
"""

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_HUB_MODEL = "victorescoto/gpt2-letras-musica-pt-br"

GENEROS = ["pop", "rap", "rock", "r&b", "country", "variado"]

st.set_page_config(page_title="Compositor IA 🎵", page_icon="🎵", layout="wide")


def resolve_model_source() -> str:
    if (LOCAL_MODEL_DIR / "config.json").exists():
        return str(LOCAL_MODEL_DIR)
    return st.secrets.get("MODEL_ID", os.environ.get("MODEL_ID", DEFAULT_HUB_MODEL))


@st.cache_resource(show_spinner="Carregando o modelo…")
def load_model():
    source = resolve_model_source()
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device, source


def build_prompt(genero: str, titulo: str, inicio: str) -> str:
    prompt = f"Gênero: {genero}\nTítulo: {titulo}\nLetra:\n"
    if inicio.strip():
        prompt += inicio.strip() + "\n"
    return prompt


st.title("🎵 Compositor IA — Letras de Música Brasileira")
st.markdown(
    "Playground de um **GPT-2 em português** fine-tunado com letras de música "
    "brasileira usando 🤗 Transformers. Escolha o gênero, dê um título e deixe "
    "o modelo compor. Ajuste os parâmetros na barra lateral para controlar a "
    "criatividade."
)

with st.sidebar:
    st.header("⚙️ Parâmetros de geração")
    temperature = st.slider(
        "Temperatura", 0.1, 1.5, 0.9, 0.05,
        help="Valores altos = mais criativo/arriscado; baixos = mais conservador.",
    )
    top_p = st.slider(
        "Top-p (nucleus)", 0.1, 1.0, 0.95, 0.05,
        help="Amostra apenas do conjunto de tokens cuja probabilidade acumulada é p.",
    )
    top_k = st.slider(
        "Top-k", 0, 100, 50, 5,
        help="Considera apenas os k tokens mais prováveis (0 desativa).",
    )
    max_new_tokens = st.slider("Tamanho máximo (tokens)", 50, 400, 220, 10)
    num_versoes = st.radio("Versões geradas", [1, 2, 3], horizontal=True)
    repeticao = st.slider(
        "Bloqueio de repetição (n-gramas)", 0, 6, 3, 1,
        help="Impede a repetição literal de sequências de n palavras.",
    )

col1, col2 = st.columns([1, 2])
with col1:
    genero = st.selectbox("Gênero musical", GENEROS, index=0)
    titulo = st.text_input("Título da música", "Coração na Contramão")
    inicio = st.text_area(
        "Começo da letra (opcional)",
        placeholder="Ex.: Hoje eu acordei pensando em você…",
        height=100,
    )
    gerar = st.button("🎤 Compor letra", type="primary", use_container_width=True)

with col2:
    if gerar:
        if not titulo.strip():
            st.warning("Dê um título para a música!")
        else:
            tokenizer, model, device, source = load_model()
            prompt = build_prompt(genero, titulo, inicio)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with st.spinner("Compondo…"):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k if top_k > 0 else None,
                        num_return_sequences=num_versoes,
                        no_repeat_ngram_size=repeticao if repeticao > 0 else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            for i, seq in enumerate(outputs, start=1):
                letra = tokenizer.decode(
                    seq[inputs.input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                st.subheader(f"Versão {i} — “{titulo}” ({genero})")
                corpo = (inicio.strip() + "\n" + letra).strip() if inicio.strip() else letra
                st.text(corpo)
                st.divider()
            st.caption(f"Modelo: `{source}` · dispositivo: `{device}`")
    else:
        st.info("Configure a música à esquerda e clique em **Compor letra**.")

st.markdown("---")
st.caption(
    "Trabalho da Fase 4 (substitutiva) — Machine Learning Engineering · "
    "Pós-graduação FIAP · Modelo base: `pierreguillou/gpt2-small-portuguese` · "
    "Dataset: letras em português do Genius Song Lyrics."
)
