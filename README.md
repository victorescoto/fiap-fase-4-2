# 🎵 Compositor IA — Gerador de Letras de Música Brasileira

Trabalho da **Prova Substitutiva — Fase 4 (Machine Learning Engineering)** da pós-graduação FIAP.

Modelo generativo construído com **Hugging Face Transformers**: um GPT-2 em português fine-tunado com letras de música em português para gerar letras originais a partir de um gênero e um título. A entrega inclui um **playground em Streamlit** para testar o poder generativo do modelo.

## 🧠 Estratégia

| Etapa | Escolha |
|---|---|
| Tema | Geração de letras de música brasileira |
| Modelo base | [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese) (GPT-2 124M pré-treinado em português) |
| Dataset | [`sebastiandizon/genius-song-lyrics`](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics), filtrado para `language == "pt"` via streaming (~12 mil letras) |
| Fine-tuning | Causal LM com `Trainer`, blocos de 512 tokens, 3 épocas, fp16, RTX 4060 |
| Avaliação | Perplexidade (base × fine-tunado), diversidade (distinct-1/2) e originalidade (sobreposição de 8-gramas com o corpus) |
| Deploy | Streamlit Community Cloud, modelo hospedado no Hugging Face Hub |

Cada letra é formatada como:

```
Gênero: pop
Título: Coração na Contramão
Letra:
<letra>
<|endoftext|>
```

Assim o modelo aprende a condicionar a geração ao gênero e ao título — que é exatamente a interface do playground.

## 📁 Estrutura

```
├── app.py                    # Playground Streamlit
├── requirements.txt          # Dependências do deploy
├── scripts/
│   ├── prepare_dataset.py    # Coleta/limpeza das letras (streaming)
│   ├── train.py              # Fine-tuning com Transformers Trainer
│   ├── evaluate.py           # Perplexidade, diversidade e originalidade
│   └── push_to_hub.py        # Publica o modelo no HF Hub
├── data/avaliacao.json       # Resultados da avaliação
└── model/                    # Modelo fine-tunado (não versionado)
```

## 🚀 Reproduzindo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt datasets

python scripts/prepare_dataset.py   # coleta ~12k letras em pt (streaming)
python scripts/train.py             # fine-tuning (~40 min em uma RTX 4060)
python scripts/evaluate.py          # métricas de qualidade e originalidade

streamlit run app.py                # playground local
```

## ☁️ Deploy no Streamlit Cloud

1. Publique o modelo no Hub: `huggingface-cli login && python scripts/push_to_hub.py`
2. No [Streamlit Community Cloud](https://share.streamlit.io), crie um app apontando para este repositório (`app.py`).
3. O app baixa o modelo do Hub automaticamente (`MODEL_ID` pode ser sobrescrito em *Secrets*).

## 📊 Avaliação

Os números ficam em [data/avaliacao.json](data/avaliacao.json) após rodar `scripts/evaluate.py`. Critérios:

- **Perplexidade** no conjunto de validação: mede o ganho do fine-tuning sobre o modelo base no domínio de letras.
- **Distinct-1/Distinct-2**: proporção de unigramas/bigramas únicos nas gerações — detecta degeneração repetitiva.
- **Sobreposição máxima de 8-gramas** com o corpus de treino: mede originalidade (valores baixos = o modelo compõe, não decora).

Os parâmetros de geração (temperatura, top-p, top-k, bloqueio de repetição) são ajustáveis no playground, permitindo explorar o compromisso entre coerência e criatividade.
