import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coliollama.api.fastapi.app import create_app
from coliollama.core.config import Settings

FAKE = Path(__file__).with_name("fake_coli.py")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        coli_command=(sys.executable, str(FAKE)),
        startup_timeout=20,
        shutdown_timeout=5,
    )


@pytest.fixture
def make_model(settings):
    def make(name):
        path = settings.models_dir / name
        path.mkdir(parents=True)
        (path / "weights.bin").write_bytes(b"x" * 100)
        return path
    return make


@pytest.fixture
def client(settings, make_model):
    make_model("alpha")
    make_model("beta")
    with TestClient(create_app(settings)) as c:
        store = c.app.state.store
        for n in ("alpha", "beta"):
            store.add(n, settings.models_dir / n)
        yield c
