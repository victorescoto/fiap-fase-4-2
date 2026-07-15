"""Fine-tuning do GPT-2 em português com letras de música brasileira.

Modelo base: pierreguillou/gpt2-small-portuguese
Formato de cada exemplo:

    Gênero: pop
    Título: Meu Coração
    Letra:
    <letra>
    <|endoftext|>

Os exemplos são concatenados e divididos em blocos de BLOCK_SIZE tokens
(causal language modeling padrão).
"""

import json
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
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "letras_pt.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "model"
BLOCK_SIZE = 512


def format_example(row: dict, eos: str) -> str:
    return (
        f"Gênero: {row['genre']}\n"
        f"Título: {row['title']}\n"
        f"Letra:\n{row['lyrics']}{eos}"
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8")]
    texts = [format_example(r, tokenizer.eos_token) for r in rows]
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
        output_dir=str(OUT_DIR / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
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

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"Modelo salvo em {OUT_DIR}")


if __name__ == "__main__":
    main()
