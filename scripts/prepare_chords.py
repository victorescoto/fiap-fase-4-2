"""Extrai progressões de acordes por gênero do dataset Chordonomicon.

Para cada música, pega os 4 primeiros acordes do primeiro refrão (ou do
primeiro verso, na ausência de refrão) e conta a frequência de cada
progressão por gênero. As mais comuns viram o pool usado para "anotar"
as letras no fine-tuning (data/progressoes.json).
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "ailsntua/Chordonomicon"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "progressoes.json"
TOP_PER_GENRE = 120

# main_genre do Chordonomicon -> gênero usado no nosso formato de treino
GENRE_MAP = {
    "pop": "pop",
    "rock": "rock",
    "rap": "rap",
    "hip hop": "rap",
    "rb": "r&b",
    "r&b": "r&b",
    "soul": "r&b",
    "country": "country",
}

SECTION_RE = re.compile(r"<(\w+?)_\d+>")
CHORD_RE = re.compile(r"^[A-G][#b]?(maj|min|m|dim|aug|sus|add)?\d*(/[A-G][#b]?)?$")


def normalize(chord: str) -> str:
    return chord.replace("min", "m").replace("maj7", "maj7")


def first_section_chords(chords_str: str, preferred: str) -> list[str] | None:
    """Retorna os 4 primeiros acordes da primeira seção `preferred`."""
    tokens = chords_str.split()
    current, grabbed = None, []
    for tok in tokens:
        m = SECTION_RE.match(tok)
        if m:
            if grabbed and current == preferred:
                break
            current = m.group(1)
            if current == preferred:
                grabbed = []
            continue
        if current == preferred and CHORD_RE.match(tok):
            grabbed.append(normalize(tok))
            if len(grabbed) == 4:
                return grabbed
    return grabbed if len(grabbed) == 4 else None


def main() -> None:
    ds = load_dataset(DATASET_ID, split="train")
    counters: dict[str, Counter] = defaultdict(Counter)

    for row in ds:
        genre = GENRE_MAP.get((row["main_genre"] or "").strip().lower())
        if genre is None:
            genre = "variado"
        chords = row["chords"] or ""
        prog = first_section_chords(chords, "chorus") or first_section_chords(
            chords, "verse"
        )
        # descarta vamps pobres (menos de 3 acordes distintos)
        if prog is None or len(set(prog)) < 3:
            continue
        counters[genre][" ".join(prog)] += 1

    pools = {
        genre: [p for p, _ in counter.most_common(TOP_PER_GENRE)]
        for genre, counter in counters.items()
    }
    # todo gênero das letras precisa de um pool; completa faltantes com pop
    for g in ["pop", "rap", "rock", "r&b", "country", "variado"]:
        if not pools.get(g):
            pools[g] = pools["pop"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(pools, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for g, ps in pools.items():
        print(f"{g}: {len(ps)} progressões (ex.: {ps[0] if ps else '-'})")
    print(f"Salvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
