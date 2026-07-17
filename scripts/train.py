"""Fine-tuning do GPT-2 em português com letras de música brasileira.

Modelo base: pierreguillou/gpt2-small-portuguese
Formato de cada exemplo (a "ficha rítmica" — BPM, compasso e acordes — é
amostrada por gênero: BPM/compasso de faixas típicas, acordes de progressões
reais do Chordonomicon extraídas por scripts/prepare_chords.py):

    Gênero: pop
    Título: Meu Coração
    BPM: 96
    Compasso: 4/4
    Acordes: C G Am F
    Letra:
    <letra>
    <|endoftext|>

Os exemplos são concatenados e divididos em blocos de BLOCK_SIZE tokens
(causal language modeling padrão).
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "pierreguillou/gpt2-small-portuguese"
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "letras_pt.jsonl"
CHORDS_PATH = ROOT / "data" / "progressoes.json"
OUT_DIR = ROOT / "model"
BLOCK_SIZE = 512

# Faixas de BPM típicas por gênero
BPM_RANGES = {
    "pop": (84, 124),
    "rap": (78, 104),
    "rock": (100, 148),
    "r&b": (68, 100),
    "country": (88, 132),
    "variado": (80, 130),
}


def sample_ficha(genre: str, pools: dict, rng: random.Random) -> tuple[int, str, str]:
    lo, hi = BPM_RANGES.get(genre, (80, 130))
    bpm = rng.randrange(lo, hi + 1, 2)
    compasso = rng.choices(
        ["4/4", "3/4", "6/8"],
        weights=[0.9, 0.06, 0.04] if genre in ("country", "variado", "r&b") else [0.97, 0.02, 0.01],
    )[0]
    acordes = rng.choice(pools.get(genre) or pools["pop"])
    return bpm, compasso, acordes


def format_example(row: dict, eos: str, pools: dict, rng: random.Random) -> str:
    bpm, compasso, acordes = sample_ficha(row["genre"], pools, rng)
    return (
        f"Gênero: {row['genre']}\n"
        f"Título: {row['title']}\n"
        f"BPM: {bpm}\n"
        f"Compasso: {compasso}\n"
        f"Acordes: {acordes}\n"
        f"Letra:\n{row['lyrics']}{eos}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="reduz VRAM (necessário p/ modelos maiores na RTX 4060)")
    p.add_argument("--optim", default="adamw_torch",
                   help="use 'adafactor' p/ modelos maiores (menos VRAM)")
    p.add_argument("--bf16", action="store_true", help="usa bf16 em vez de fp16")
    return p.parse_args()


def main() -> None:
    args_cli = parse_args()
    out_dir = Path(args_cli.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args_cli.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args_cli.base_model)
    if args_cli.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8")]
    pools = json.loads(CHORDS_PATH.read_text(encoding="utf-8"))
    rng = random.Random(42)
    texts = [format_example(r, tokenizer.eos_token, pools, rng) for r in rows]
    ds = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.05, seed=42)

    def tokenize(batch):
        return tokenizer(batch["text"])

    def group_texts(batch):
        ids = [tok for seq in batch["input_ids"] for tok in seq]
        total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
        chunks = [ids[i : i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
        return {"input_ids": chunks, "attention_mask": [[1] * BLOCK_SIZE] * len(chunks)}

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    lm_ds = tokenized.map(group_texts, batched=True, batch_size=1000)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args_cli.epochs,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_accum,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim=args_cli.optim,
        fp16=not args_cli.bf16,
        bf16=args_cli.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Métricas finais:", metrics)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Modelo salvo em {out_dir}")


if __name__ == "__main__":
    main()
