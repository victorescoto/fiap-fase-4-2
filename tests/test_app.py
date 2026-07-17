import importlib

import app


def test_app_importa_em_modo_bare():
    # o script inteiro roda fora do runtime do Streamlit sem exceção
    importlib.reload(app)


def test_ficha_re_extrai_os_tres_campos():
    saida = "BPM: 96\nCompasso: 4/4\nAcordes: C G Am F\nLetra:\nMeu verso"
    m = app.FICHA_RE.search(saida)
    assert m is not None
    assert m["bpm"] == "96"
    assert m["compasso"] == "4/4"
    assert m["acordes"] == "C G Am F"


def test_ficha_re_rejeita_saida_malformada():
    assert app.FICHA_RE.search("BPM: rápido\nCompasso: 4/4\nLetra:\n") is None
    assert app.FICHA_RE.search("Letra:\nsem ficha nenhuma") is None


def test_resolve_model_source_ignora_pasta_local_sem_pesos(tmp_path, monkeypatch):
    # config.json sem model.safetensors (o caso do deploy que quebrou):
    # o app deve ir para o Hub, não para a pasta local incompleta
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(app, "LOCAL_MODEL_DIR", tmp_path)
    monkeypatch.delenv("MODEL_ID", raising=False)
    assert app.resolve_model_source() == app.DEFAULT_HUB_MODEL


def test_resolve_model_source_usa_pasta_local_completa(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(app, "LOCAL_MODEL_DIR", tmp_path)
    assert app.resolve_model_source() == str(tmp_path)


def test_resolve_model_source_respeita_variavel_de_ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "LOCAL_MODEL_DIR", tmp_path)
    monkeypatch.setenv("MODEL_ID", "outro/modelo")
    assert app.resolve_model_source() == "outro/modelo"
