from prepare_chords import CHORD_RE, first_section_chords, normalize


def test_prefere_refrao_ao_verso():
    chords = "<verse_1> C G Am F <chorus_1> D A Bm G"
    assert first_section_chords(chords, "chorus") == ["D", "A", "Bm", "G"]


def test_usa_verso_quando_nao_ha_refrao():
    chords = "<intro_1> E <verse_1> C G Am F C G"
    assert first_section_chords(chords, "chorus") is None
    assert first_section_chords(chords, "verse") == ["C", "G", "Am", "F"]


def test_retorna_none_com_menos_de_4_acordes():
    assert first_section_chords("<chorus_1> C G", "chorus") is None


def test_normaliza_min_para_m():
    assert normalize("Amin") == "Am"
    chords = "<chorus_1> Amin C G Emin"
    assert first_section_chords(chords, "chorus") == ["Am", "C", "G", "Em"]


def test_ignora_tokens_que_nao_sao_acordes():
    assert CHORD_RE.match("C#m7") is not None
    assert CHORD_RE.match("G/B") is not None
    assert CHORD_RE.match("N.C.") is None
    assert CHORD_RE.match("x2") is None
    chords = "<chorus_1> C x2 G N.C. Am F"
    assert first_section_chords(chords, "chorus") == ["C", "G", "Am", "F"]


def test_para_na_proxima_secao():
    # o refrão tem só 3 acordes; não deve completar com os da ponte
    chords = "<chorus_1> C G Am <bridge_1> F D E B"
    assert first_section_chords(chords, "chorus") is None
