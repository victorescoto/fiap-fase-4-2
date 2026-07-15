"""Coleta letras de música em português do dataset Genius Song Lyrics.

Usa streaming para não baixar os 9 GB do CSV completo: percorre as linhas,
filtra `language == "pt"`, limpa a letra e salva em data/letras_pt.jsonl.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "sebastiandizon/genius-song-lyrics"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "letras_pt.jsonl"
TARGET_SONGS = 12_000
MAX_ROWS_SCANNED = 2_500_000
MIN_CHARS, MAX_CHARS = 200, 4_000

# Mapeia os gêneros do Genius para nomes em português usados no prompt
TAG_PT = {
    "pop": "pop",
    "rap": "rap",
    "rock": "rock",
    "rb": "r&b",
    "country": "country",
    "misc": "variado",
}

SECTION_RE = re.compile(r"^\[[^\]]*\]\s*$")  # linhas tipo [Refrão], [Verse 1]
EMBED_RE = re.compile(r"\d*Embed\s*$")
BLANKS_RE = re.compile(r"\n{3,}")


def clean_lyrics(text: str) -> str | None:
    text = EMBED_RE.sub("", text.strip())
    # remove a primeira linha se for o cabeçalho "... Lyrics" do Genius
    lines = text.splitlines()
    if lines and lines[0].strip().lower().endswith("lyrics"):
        lines = lines[1:]
    lines = [l for l in lines if not SECTION_RE.match(l.strip())]
    text = BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    return text


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    kept, scanned = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in ds:
            scanned += 1
            if scanned > MAX_ROWS_SCANNED or kept >= TARGET_SONGS:
                break
            if row.get("language") != "pt":
                continue
            lyrics = clean_lyrics(row.get("lyrics") or "")
            if lyrics is None:
                continue
            fh.write(
                json.dumps(
                    {
                        "title": (row.get("title") or "").strip(),
                        "artist": (row.get("artist") or "").strip(),
                        "genre": TAG_PT.get(row.get("tag"), "variado"),
                        "lyrics": lyrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if kept % 500 == 0:
                print(f"{kept} letras coletadas ({scanned} linhas lidas)", flush=True)

    print(f"Concluído: {kept} letras salvas em {OUT_PATH} ({scanned} linhas lidas)")


if __name__ == "__main__":
    main()
