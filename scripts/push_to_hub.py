"""Publica o modelo fine-tunado no Hugging Face Hub.

Pré-requisito: `huggingface-cli login` (token com permissão de escrita).
Uso: python scripts/push_to_hub.py [repo_id] [pasta_do_modelo]
Ex.:  python scripts/push_to_hub.py victorescoto/tucano-630m-letras-pt-br model_tucano
"""

import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "victorescoto/gpt2-letras-musica-pt-br"


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    model_dir = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "model")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Modelo publicado em https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
