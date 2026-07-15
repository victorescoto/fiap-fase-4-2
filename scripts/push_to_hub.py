"""Publica o modelo fine-tunado no Hugging Face Hub.

Pré-requisito: `huggingface-cli login` (token com permissão de escrita).
Uso: python scripts/push_to_hub.py [repo_id]
"""

import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DEFAULT_REPO = "victorescoto/gpt2-letras-musica-pt-br"


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Modelo publicado em https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
