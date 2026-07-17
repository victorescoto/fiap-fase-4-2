from prepare_dataset import MAX_CHARS, MIN_CHARS, TAG_PT, clean_lyrics

# corpo longo o suficiente para passar no filtro de tamanho mínimo
VERSO = "Hoje o sol nasceu pra nós dois e a cidade acordou cantando\n"
CORPO = VERSO * 8


def test_remove_marcadores_de_secao():
    texto = f"[Verse 1]\n{CORPO}[Refrão]\n{CORPO}"
    limpo = clean_lyrics(texto)
    assert limpo is not None
    assert "[Verse 1]" not in limpo
    assert "[Refrão]" not in limpo
    assert VERSO.strip() in limpo


def test_remove_sufixo_embed_do_genius():
    limpo = clean_lyrics(CORPO + "123Embed")
    assert limpo is not None
    assert "Embed" not in limpo


def test_remove_cabecalho_lyrics_da_primeira_linha():
    texto = "Minha Canção Lyrics\n" + CORPO
    limpo = clean_lyrics(texto)
    assert limpo is not None
    assert "Lyrics" not in limpo.splitlines()[0]


def test_colapsa_linhas_em_branco_multiplas():
    limpo = clean_lyrics(CORPO + "\n\n\n\n" + CORPO)
    assert limpo is not None
    assert "\n\n\n" not in limpo


def test_rejeita_letra_curta_demais():
    assert clean_lyrics("la la la") is None
    assert len("la la la") < MIN_CHARS


def test_rejeita_letra_longa_demais():
    assert clean_lyrics(VERSO * 200) is None
    assert len(VERSO * 200) > MAX_CHARS


def test_mapa_de_generos_cobre_os_seis_do_projeto():
    assert set(TAG_PT.values()) == {"pop", "rap", "rock", "r&b", "country", "variado"}
