"""Playground de geração de letras de música brasileira.

Carrega o GPT-2 em português fine-tunado com letras de música e permite
experimentar os parâmetros de geração (temperatura, top-p, top-k etc.).
Além da letra, o modelo gera a "ficha rítmica" da música: BPM, compasso
e progressão de acordes (aprendidos no fine-tuning a partir do
Chordonomicon + faixas de BPM típicas por gênero).
"""

import os
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_HUB_MODEL = "victorescoto/gpt2-letras-musica-pt-br"

GENEROS = ["pop", "rap", "rock", "r&b", "country", "variado"]

FICHA_RE = re.compile(
    r"BPM:\s*(?P<bpm>\d+)\s*\nCompasso:\s*(?P<compasso>[\d/]+)\s*\n"
    r"Acordes:\s*(?P<acordes>[^\n]+)\nLetra:\n?",
)

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


@torch.no_grad()
def sample(tokenizer, model, device, prompt: str, params: dict, max_new: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"] if params["top_k"] > 0 else None,
        no_repeat_ngram_size=params["repeticao"] if params["repeticao"] > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def compor(tokenizer, model, device, genero, titulo, inicio, params):
    """Gera a ficha rítmica e a letra. Retorna (ficha | None, letra)."""
    base = f"Gênero: {genero}\nTítulo: {titulo}\n"

    # Etapa 1: ficha rítmica (curta; termina em "Letra:")
    ficha = None
    for _ in range(2):  # uma nova tentativa se a ficha vier malformada
        m = FICHA_RE.search(sample(tokenizer, model, device, base, params, 40))
        if m:
            ficha = m.groupdict()
            break
    ficha_txt = (
        f"BPM: {ficha['bpm']}\nCompasso: {ficha['compasso']}\n"
        f"Acordes: {ficha['acordes']}\n"
        if ficha
        else ""
    )

    # Etapa 2: letra, condicionada à ficha (e ao começo dado pelo usuário)
    prompt = base + ficha_txt + "Letra:\n"
    if inicio.strip():
        prompt += inicio.strip() + "\n"
    letra = sample(tokenizer, model, device, prompt, params, params["max_new_tokens"])
    if inicio.strip():
        letra = inicio.strip() + "\n" + letra
    return ficha, letra.strip()


st.title("🎵 Compositor IA — Letras de Música Brasileira")
st.markdown(
    "Playground de um **GPT-2 em português** fine-tunado com letras de música "
    "brasileira usando 🤗 Transformers. Escolha o gênero, dê um título e deixe o "
    "modelo compor **a letra e a ficha rítmica** (BPM, compasso e acordes). "
    "Ajuste os parâmetros na barra lateral para controlar a criatividade."
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

params = {
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "max_new_tokens": max_new_tokens,
    "repeticao": repeticao,
}

col1, col2 = st.columns([1, 2])
with col1:
    genero = st.selectbox("Gênero musical", GENEROS, index=0)
    titulo = st.text_input("Título da música", "Coração na Contramão")
    inicio = st.text_area(
        "Começo da letra (opcional)",
        placeholder="Ex.: Hoje eu acordei pensando em você…",
        height=100,
    )
    gerar = st.button("🎤 Compor música", type="primary", use_container_width=True)

with col2:
    if gerar:
        if not titulo.strip():
            st.warning("Dê um título para a música!")
        else:
            tokenizer, model, device, source = load_model()
            for i in range(1, num_versoes + 1):
                with st.spinner(f"Compondo versão {i}…"):
                    ficha, letra = compor(
                        tokenizer, model, device, genero, titulo, inicio, params
                    )
                st.subheader(f"Versão {i} — “{titulo}” ({genero})")
                if ficha:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🥁 BPM", ficha["bpm"])
                    c2.metric("🎼 Compasso", ficha["compasso"])
                    c3.metric("🎸 Acordes", ficha["acordes"])
                st.text(letra)
                st.divider()
            st.caption(f"Modelo: `{source}` · dispositivo: `{device}`")
    else:
        st.info("Configure a música à esquerda e clique em **Compor música**.")

st.markdown("---")
st.caption(
    "Trabalho da Fase 4 (substitutiva) — Machine Learning Engineering · "
    "Pós-graduação FIAP · Modelo base: `pierreguillou/gpt2-small-portuguese` · "
    "Datasets: letras em português do Genius Song Lyrics + progressões do "
    "Chordonomicon."
)
