"""Avaliação do modelo fine-tunado vs. modelo base.

Métricas:
1. Perplexidade no conjunto de validação (base vs. fine-tunado);
2. Diversidade das gerações (distinct-1 / distinct-2);
3. Originalidade: maior sobreposição de n-gramas (8-gramas) entre o texto
   gerado e o corpus de treino — valores baixos indicam que o modelo não
   está decorando letras.
"""

import json
import math
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "pierreguillou/gpt2-small-portuguese"
ROOT = Path(__file__).resolve().parent.parent
FT_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "letras_pt.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    ("pop", "Coração na Contramão"),
    ("rap", "Vida no Corre"),
    ("rock", "Estrada Sem Fim"),
    ("pop", "Saudade de Você"),
    ("variado", "Domingo de Sol"),
]


FICHA_RE = re.compile(
    r"BPM:\s*\d+\s*\nCompasso:\s*[\d/]+\s*\nAcordes:\s*[^\n]+\nLetra:\n?"
)


def build_prompt(genre: str, title: str) -> str:
    # o modelo fine-tunado completa com a ficha rítmica (BPM/compasso/acordes)
    # e depois a letra; o modelo base completa texto livre
    return f"Gênero: {genre}\nTítulo: {title}\n"


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str]) -> float:
    model.eval()
    nll, n_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.input_ids.to(DEVICE)
        out = model(input_ids, labels=input_ids)
        n = input_ids.size(1) - 1
        nll += out.loss.item() * n
        n_tokens += n
    return math.exp(nll / n_tokens)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, n: int = 2) -> list[str]:
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        num_return_sequences=n,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(seq[enc.input_ids.size(1) :], skip_special_tokens=True)
        for seq in out
    ]


def distinct_n(texts: list[str], n: int) -> float:
    all_ngrams, uniq = 0, set()
    for t in texts:
        tokens = t.split()
        grams = list(zip(*[tokens[i:] for i in range(n)]))
        all_ngrams += len(grams)
        uniq.update(grams)
    return len(uniq) / all_ngrams if all_ngrams else 0.0


def max_ngram_overlap(generated: str, corpus_ngrams: set, n: int = 8) -> float:
    tokens = generated.split()
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    if not grams:
        return 0.0
    hits = sum(1 for g in grams if g in corpus_ngrams)
    return hits / len(grams)


def main() -> None:
    random.seed(42)
    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8")]
    random.shuffle(rows)
    val_texts = [r["lyrics"] for r in rows[:200]]

    corpus_ngrams: set = set()
    for r in rows[200:5200]:
        tokens = r["lyrics"].split()
        corpus_ngrams.update(zip(*[tokens[i:] for i in range(8)]))

    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    results = {}

    for name, source in [("base", BASE_MODEL), ("fine-tunado", FT_DIR)]:
        model = AutoModelForCausalLM.from_pretrained(source).to(DEVICE)
        ppl = perplexity(model, tokenizer, val_texts)
        gens = []
        for genre, title in PROMPTS:
            gens.extend(generate(model, tokenizer, build_prompt(genre, title)))
        overlap = max(max_ngram_overlap(g, corpus_ngrams) for g in gens)
        ficha_ok = sum(1 for g in gens if FICHA_RE.search(g)) / len(gens)
        results[name] = {
            "perplexidade": round(ppl, 2),
            "distinct-1": round(distinct_n(gens, 1), 3),
            "distinct-2": round(distinct_n(gens, 2), 3),
            "sobreposicao_8gramas_max": round(overlap, 3),
            "ficha_ritmica_valida": round(ficha_ok, 2),
        }
        print(f"\n===== Modelo {name} =====")
        for k, v in results[name].items():
            print(f"  {k}: {v}")
        print("\n--- Amostra ---")
        print(f"[{PROMPTS[0][0]} / {PROMPTS[0][1]}]\n{gens[0][:600]}")
        del model
        torch.cuda.empty_cache()

    out = ROOT / "data" / "avaliacao.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {out}")


if __name__ == "__main__":
    main()
