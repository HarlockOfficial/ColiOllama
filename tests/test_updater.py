import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

from coliollama.core.config import Settings
from coliollama.core.engine import gpu
from coliollama.core.engine import installer as inst
from coliollama.core.engine.installer import EngineInstaller, UpdateError, parse_version, safe_extract

ASSET = f"colibri-v2.0.0-{inst._platform_tag()}.tar.gz"


def make_tar(version="2.0.0", extra=None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        files = {"coli": "import sys\nsys.exit(0)\n", "version.py": f'__version__ = "{version}"\n'}
        files.update(extra or {})
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_installer(tmp_path, tar: bytes, sums_ok=True, tag="v2.0.0") -> EngineInstaller:
    digest = hashlib.sha256(tar).hexdigest() if sums_ok else "0" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/releases"):
            return httpx.Response(200, json=[
                {"tag_name": "v3.0.0-rc1", "prerelease": True, "assets": []},
                {"tag_name": tag, "prerelease": False, "assets": [
                    {"name": ASSET, "browser_download_url": "https://x/asset"},
                    {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums"},
                ], "tarball_url": "https://x/src"},
            ])
        if url == "https://x/asset":
            return httpx.Response(200, content=tar)
        if url == "https://x/sums":
            return httpx.Response(200, text=f"{digest}  {ASSET}\n")
        return httpx.Response(404)

    settings = Settings(home=tmp_path / "home")
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return EngineInstaller(settings, client=client, api="https://api/releases")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("COLIOLLAMA_COLI", "COLIBRI_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir("/")
    monkeypatch.setattr(inst.EngineInstaller, "gpu_build_possible", lambda self: False)


def test_parse_version():
    assert parse_version("v1.12.1") > parse_version("1.9.0")
    assert parse_version("2.0.0") == parse_version("2.0.0-rc1")


def test_install_makes_new_version_the_main_engine(tmp_path):
    installer = make_installer(tmp_path, make_tar())
    messages = []
    assert installer.update(say=messages.append) == "2.0.0"
    settings = installer.settings
    cmd = settings.resolve_coli()
    assert cmd[-1] == str(settings.engines_dir / "versions" / "2.0.0" / "coli")
    assert installer.current_version() == "2.0.0"
    # already current: nothing happens
    assert installer.update(say=messages.append) is None
    assert installer.check()[0].update_available is False


def test_old_install_kept_as_previous_and_older_pruned(tmp_path):
    installer = make_installer(tmp_path, make_tar())
    versions = installer.engines / "versions"
    for old in ("0.8.0", "0.9.0"):
        (versions / old).mkdir(parents=True)
        (versions / old / "coli").write_text("")
        installer._activate(old, versions / old)
    installer.update(say=lambda m: None)
    assert sorted(p.name for p in versions.iterdir()) == ["0.9.0", "2.0.0"]
    assert inst.read_current(installer.engines)["previous"] == "0.9.0"


def test_checksum_mismatch_is_rejected(tmp_path):
    installer = make_installer(tmp_path, make_tar(), sums_ok=False)
    with pytest.raises(UpdateError, match="checksum"):
        installer.update(say=lambda m: None)
    assert inst.read_current(installer.engines) is None
    assert not list(installer.engines.glob(".staging-*"))


def test_version_mismatch_is_rejected(tmp_path):
    installer = make_installer(tmp_path, make_tar(version="1.0.0"))
    with pytest.raises(UpdateError, match="does not match"):
        installer.update(say=lambda m: None)


def test_unsafe_archive_rejected(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("../evil")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    archive = tmp_path / "a.tar"
    archive.write_bytes(buf.getvalue())
    with pytest.raises(UpdateError, match="unsafe"):
        safe_extract(archive, tmp_path / "out")


def test_pinned_engine_is_not_managed(tmp_path, monkeypatch):
    monkeypatch.setenv("COLIOLLAMA_COLI", f"{sys.executable} /x/coli")
    assert Settings(home=tmp_path).engine_pinned
    assert Settings(home=tmp_path).resolve_coli() == (sys.executable, "/x/coli")


def test_gpu_args(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_usable", lambda command, path: path == "/gpu")
    assert gpu.engine_gpu_args("auto", ("coli",), "/gpu", ()) == (["--gpu", "auto", "--auto-tier"], "gpu")
    assert gpu.engine_gpu_args("auto", ("coli",), "/cpu", ()) == ([], "cpu")
    assert gpu.engine_gpu_args("none", ("coli",), "/gpu", ()) == (["--gpu", "none"], "cpu")
    assert gpu.engine_gpu_args("0,1", ("coli",), "/cpu", ())[0][:2] == ["--gpu", "0,1"]
    assert gpu.engine_gpu_args("auto", ("coli",), "/gpu", ("--gpu", "none")) == ([], "custom")
