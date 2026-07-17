import random

from train import BPM_RANGES, format_example, sample_ficha

POOLS = {"pop": ["C G Am F", "F C G Am"], "rap": ["Em C G D"]}


def test_bpm_dentro_da_faixa_do_genero():
    rng = random.Random(42)
    for _ in range(50):
        bpm, _, _ = sample_ficha("pop", POOLS, rng)
        lo, hi = BPM_RANGES["pop"]
        assert lo <= bpm <= hi


def test_compasso_e_valido():
    rng = random.Random(42)
    compassos = {sample_ficha("pop", POOLS, rng)[1] for _ in range(200)}
    assert compassos <= {"4/4", "3/4", "6/8"}
    assert "4/4" in compassos  # o mais comum tem que aparecer


def test_acordes_vem_do_pool_do_genero():
    rng = random.Random(42)
    for _ in range(20):
        _, _, acordes = sample_ficha("rap", POOLS, rng)
        assert acordes == "Em C G D"


def test_genero_desconhecido_usa_fallback():
    rng = random.Random(42)
    bpm, _, acordes = sample_ficha("forró", POOLS, rng)
    assert acordes in POOLS["pop"]
    assert 80 <= bpm <= 130


def test_amostragem_e_deterministica_com_mesma_seed():
    a = [sample_ficha("pop", POOLS, random.Random(7)) for _ in range(5)]
    b = [sample_ficha("pop", POOLS, random.Random(7)) for _ in range(5)]
    assert a == b


def test_formato_do_exemplo_de_treino():
    row = {
        "genre": "pop",
        "title": "Coração na Contramão",
        "lyrics": "Meu coração\nBate na contramão",
    }
    texto = format_example(row, "<|endoftext|>", POOLS, random.Random(42))
    linhas = texto.splitlines()
    assert linhas[0] == "Gênero: pop"
    assert linhas[1] == "Título: Coração na Contramão"
    assert linhas[2].startswith("BPM: ")
    assert linhas[3].startswith("Compasso: ")
    assert linhas[4].startswith("Acordes: ")
    assert linhas[5] == "Letra:"
    assert texto.endswith("Bate na contramão<|endoftext|>")
