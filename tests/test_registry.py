import pytest

from coliollama.registry.huggingface_resolver import (
    HuggingFaceResolver,
    ModelResolutionError,
    looks_like_repo_id,
)
from coliollama.registry.local_store import LocalStore


def test_repo_id_detection():
    assert looks_like_repo_id("org/repo-name.v1")
    for bad in ("plain", "../x", "a/../b", "/abs/path", "a/b/c", "./a/b"):
        assert not looks_like_repo_id(bad)


def _download(repo_id, local_dir, revision, ready=False, token=None):
    import pathlib
    pathlib.Path(local_dir).mkdir(parents=True)
    pathlib.Path(local_dir, "w.bin").write_bytes(b"abc")
    if ready:
        pathlib.Path(local_dir, "ready").write_text("")


def test_ensure_downloads_once_and_registers(settings):
    calls = []

    def fake_download(repo_id, local_dir, revision, token=None):
        calls.append((repo_id, local_dir, revision))
        import pathlib
        pathlib.Path(local_dir).mkdir(parents=True)
        pathlib.Path(local_dir, "w.bin").write_bytes(b"abc")
        return local_dir

    store = LocalStore(settings.registry_path)
    res = HuggingFaceResolver(store, settings, fake_download)
    entry = res.ensure("org/model", revision="main", convert=False)
    assert not entry.verified and entry.size_bytes() == 3 and entry.path.endswith("org--model")
    assert res.ensure("org/model", convert=False).path == entry.path and len(calls) == 1
    assert store.get("org/model:latest") is not None


def test_ensure_local_dir_and_errors(settings, tmp_path):
    d = tmp_path / "mymodel"
    d.mkdir()
    res = HuggingFaceResolver(LocalStore(settings.registry_path), settings, lambda **k: 1 / 0)
    assert res.ensure(str(d), convert=False).name == "mymodel"
    with pytest.raises(ModelResolutionError, match="not registered"):
        res.ensure("justaname")
    with pytest.raises(ModelResolutionError, match="pull"):
        res.ensure("org/x", allow_download=False)
    with pytest.raises(ModelResolutionError, match="download of"):
        res.ensure("org/x", convert=False)


def _resolver(settings, ready):
    store = LocalStore(settings.registry_path)
    return store, HuggingFaceResolver(store, settings, lambda **kw: _download(ready=ready, **kw))


def test_usable_download_is_verified_not_converted(settings):
    store, res = _resolver(settings, ready=True)
    msgs = []
    entry = res.ensure("org/m", on_status=msgs.append)
    assert entry.verified and entry.path.endswith("org--m")
    assert any("not a model problem" in m for m in msgs)  # engine.binary failure is only a warning
    assert not res.converted_dir_for("org/m").exists()


def test_unusable_download_is_converted_and_source_removed(settings):
    store, res = _resolver(settings, ready=False)
    entry = res.ensure("org/m")
    assert entry.verified and entry.path.endswith("org--m-coli") and entry.name == "org/m"
    assert (res.converted_dir_for("org/m") / "ready").exists()
    assert not res.local_dir_for("org/m").exists()
    assert store.get("org/m").path == entry.path


def test_keep_source_and_no_convert(settings):
    _, res = _resolver(settings, ready=False)
    res.ensure("org/m", keep_source=True)
    assert res.local_dir_for("org/m").exists()
    _, res2 = _resolver(settings, ready=False)
    entry = res2.ensure("org/n", convert=False)
    assert not entry.verified and entry.path.endswith("org--n")


def test_convert_failure_keeps_raw_and_registers_nothing(settings, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_RC", "3")
    store, res = _resolver(settings, ready=False)
    with pytest.raises(ModelResolutionError, match="exit code 3"):
        res.ensure("org/m")
    assert store.get("org/m") is None and res.local_dir_for("org/m").exists()


def test_unusable_local_dir_cannot_be_converted(settings, tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    store = LocalStore(settings.registry_path)
    res = HuggingFaceResolver(store, settings)
    with pytest.raises(ModelResolutionError, match="not directly usable"):
        res.ensure(str(d))
    assert store.list() == []
    (d / "ready").write_text("")
    assert res.ensure(str(d)).verified


def test_registered_unverified_is_checked_once(settings, tmp_path):
    d = tmp_path / "ok"
    d.mkdir()
    (d / "ready").write_text("")
    store = LocalStore(settings.registry_path)
    store.add("ok", d)
    res = HuggingFaceResolver(store, settings)
    assert res.ensure("ok").verified and store.get("ok").verified
    (d / "ready").unlink()
    assert res.ensure("ok").verified  # verified entries are not re-checked


def test_doctor_classification():
    from coliollama.registry.readiness import parse_doctor_report

    r = parse_doctor_report({"checks": [
        {"id": "config.arguments", "status": "fail", "summary": "cannot read config.json"},
        {"id": "engine.binary", "status": "fail", "summary": "not built"},
        {"id": "model.shards", "status": "warn", "summary": "meh"},
    ]})
    assert not r.usable and len(r.model_problems) == 1 and len(r.environment_problems) == 1


def test_smoke_test_catches_what_doctor_misses(settings, monkeypatch):
    """Raw checkpoints pass doctor but crash on the first request; conversion must still trigger."""
    monkeypatch.setenv("FAKE_REQUIRE_READY", "1")
    store, res = _resolver(settings, ready=False)
    entry = res.ensure("org/m")
    assert entry.path.endswith("org--m-coli") and entry.verified


def test_smoke_test_passes_ready_model(settings, monkeypatch):
    monkeypatch.setenv("FAKE_REQUIRE_READY", "1")
    store, res = _resolver(settings, ready=True)
    assert res.ensure("org/m").path.endswith("org--m")


def test_hf_token_from_dotenv(tmp_path, monkeypatch):
    from coliollama.core.config import Settings

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('# c\nexport HF_TOKEN="hf_abc"\n')
    assert Settings(home=tmp_path / "h").hf_token == "hf_abc"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert Settings(home=tmp_path / "h").hf_token == "hf_env"
