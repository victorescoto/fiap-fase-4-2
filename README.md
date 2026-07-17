# 🎵 Compositor IA — Gerador de Letras de Música Brasileira

Trabalho da **Prova Substitutiva — Fase 4 (Machine Learning Engineering)** da pós-graduação FIAP.

Modelo generativo construído com **Hugging Face Transformers**: um GPT-2 em português fine-tunado com letras de música em português para gerar, a partir de um gênero e um título, uma **letra original** e a **ficha rítmica** da música (BPM, compasso e progressão de acordes). A entrega inclui um **playground em Streamlit** para testar o poder generativo do modelo.

## 🧠 Estratégia

| Etapa | Escolha |
|---|---|
| Tema | Geração de letras de música brasileira |
| Modelo base | [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese) (GPT-2 124M pré-treinado em português) |
| Datasets | Letras: [`sebastiandizon/genius-song-lyrics`](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics), filtrado para `language == "pt"` via streaming (~12 mil letras) · Acordes: [`ailsntua/Chordonomicon`](https://huggingface.co/datasets/ailsntua/Chordonomicon) (progressões reais por gênero) |
| Fine-tuning | Causal LM com `Trainer`, blocos de 512 tokens, 3 épocas, fp16, RTX 4060 |
| Avaliação | Perplexidade (base × fine-tunado), diversidade (distinct-1/2) e originalidade (sobreposição de 8-gramas com o corpus) |
| Deploy | Streamlit Community Cloud, modelo hospedado no Hugging Face Hub |

Cada letra é formatada como:

```
Gênero: pop
Título: Coração na Contramão
BPM: 96
Compasso: 4/4
Acordes: C G Am F
Letra:
<letra>
<|endoftext|>
```

Assim o modelo aprende a condicionar a geração ao gênero e ao título — que é exatamente a interface do playground. A **ficha rítmica** de cada exemplo de treino é amostrada por gênero: BPM e compasso de faixas típicas do gênero, acordes de progressões reais extraídas do Chordonomicon (só progressões com 3+ acordes distintos, as mais frequentes de cada gênero). Na geração, o modelo compõe primeiro a ficha e depois a letra condicionada a ela.

## 📁 Estrutura

```
├── app.py                    # Playground Streamlit
├── requirements.txt          # Dependências do deploy
├── scripts/
│   ├── prepare_dataset.py    # Coleta/limpeza das letras (streaming)
│   ├── prepare_chords.py     # Progressões de acordes por gênero (Chordonomicon)
│   ├── train.py              # Fine-tuning com Transformers Trainer
│   ├── evaluate.py           # Perplexidade, diversidade e originalidade
│   └── push_to_hub.py        # Publica o modelo no HF Hub
├── tests/                    # Testes unitários (pytest)
├── data/progressoes.json     # Pools de progressões por gênero
├── data/avaliacao.json       # Resultados da avaliação
└── model/                    # Modelo fine-tunado (não versionado)
```

## 🚀 Reproduzindo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt datasets

python scripts/prepare_dataset.py   # coleta ~12k letras em pt (streaming)
python scripts/prepare_chords.py    # progressões de acordes por gênero
python scripts/train.py             # fine-tuning (~25 min em uma RTX 4060)
python scripts/evaluate.py          # métricas de qualidade e originalidade

streamlit run app.py                # playground local
```

### Testes

```bash
pip install pytest
pytest
```

Cobrem a limpeza das letras, a extração de progressões do Chordonomicon, a amostragem/formato da ficha rítmica no treino, o parsing da ficha gerada e a resolução da origem do modelo no app (pasta local completa × HF Hub × variável `MODEL_ID`).

## ☁️ Deploy no Streamlit Cloud

1. Publique o modelo no Hub: `huggingface-cli login && python scripts/push_to_hub.py`
2. No [Streamlit Community Cloud](https://share.streamlit.io), crie um app apontando para este repositório (`app.py`).
3. O app baixa o modelo do Hub automaticamente (`MODEL_ID` pode ser sobrescrito em *Secrets*).

## 📊 Avaliação

Os números ficam em [data/avaliacao.json](data/avaliacao.json) após rodar `scripts/evaluate.py`. Critérios:

- **Perplexidade** no conjunto de validação: mede o ganho do fine-tuning sobre o modelo base no domínio de letras.
- **Distinct-1/Distinct-2**: proporção de unigramas/bigramas únicos nas gerações — detecta degeneração repetitiva.
- **Sobreposição máxima de 8-gramas** com o corpus de treino: mede originalidade (valores baixos = o modelo compõe, não decora).
- **Ficha rítmica válida**: fração das gerações em que o modelo produz BPM/compasso/acordes no formato esperado.

Os parâmetros de geração (temperatura, top-p, top-k, bloqueio de repetição) são ajustáveis no playground, permitindo explorar o compromisso entre coerência e criatividade.
