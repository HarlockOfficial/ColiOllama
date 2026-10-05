"""Find, download, unpack and activate Colibrí releases (and build the CUDA engines from source).

Layout under `<home>/engines/`:
    versions/<x.y.z>/   unpacked release (flat: `coli`, engine binaries, version.py, ...)
    current.json        {"version", "path", "previous"}: the install used as the main engine
    state.json          bookkeeping for the background checker
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from coliollama.core.config import Settings
from coliollama.core.engine import gpu

log = logging.getLogger("coliollama.update")

RELEASES_API = "https://api.github.com/repos/JustVugg/colibri/releases"
Say = Callable[[str], None]


class UpdateError(Exception):
    pass


def parse_version(text: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", text.split("-")[0].split("+")[0])
    return tuple(int(n) for n in nums[:3]) or (0,)


def _platform_tag() -> str | None:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
        return "macos-arm64"
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x86_64"
    return None


# ---- managed install pointer ------------------------------------------------------------

def _current_file(engines_dir: Path) -> Path:
    return engines_dir / "current.json"


def read_current(engines_dir: Path) -> dict | None:
    try:
        return json.loads(_current_file(engines_dir).read_text())
    except (OSError, ValueError):
        return None


def managed_root(engines_dir: Path) -> Path | None:
    cur = read_current(engines_dir)
    if cur and cur.get("path") and (Path(cur["path"]) / "coli").is_file():
        return Path(cur["path"])
    return None


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _read_state(engines_dir: Path) -> dict:
    try:
        return json.loads((engines_dir / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def _write_state(engines_dir: Path, **updates) -> None:
    state = _read_state(engines_dir)
    state.update(updates)
    _write_json_atomic(engines_dir / "state.json", state)


@contextlib.contextmanager
def _install_lock(engines_dir: Path):
    engines_dir.mkdir(parents=True, exist_ok=True)
    with open(engines_dir / ".lock", "w") as fh:
        try:
            import fcntl
        except ImportError:  # Windows: no cross-process lock
            yield
            return
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise UpdateError("another update is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---- archive handling ----------------------------------------------------------------------

def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar/zip refusing absolute paths, `..`, links and device files."""
    dest = dest.resolve()

    def check(name: str) -> None:
        target = (dest / name).resolve()
        if Path(name).is_absolute() or dest != target and dest not in target.parents:
            raise UpdateError(f"unsafe path in archive: {name}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                check(name)
            zf.extractall(dest)
        return
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            check(member.name)
            if not (member.isfile() or member.isdir()):
                raise UpdateError(f"unsupported entry in archive: {member.name}")
        tf.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(dest)  # validated above


def _flatten_root(directory: Path) -> Path:
    """Return the directory that actually holds `coli` (flat layout or a single top directory)."""
    if (directory / "coli").is_file():
        return directory
    entries = [p for p in directory.iterdir() if p.is_dir()]
    for entry in entries:
        for sub in (entry, entry / "c"):
            if (sub / "coli").is_file():
                return sub
    raise UpdateError("archive does not contain a `coli` launcher")


# ---- releases ---------------------------------------------------------------------------------

@dataclass
class Release:
    tag: str
    version: str
    asset_url: str | None
    asset_name: str | None
    sums_url: str | None
    source_url: str | None
    prerelease: bool = False


@dataclass
class UpdateStatus:
    current: str | None
    latest: str | None
    update_available: bool
    gpu_build_needed: bool
    managed: bool
    pinned: bool


def release_from_json(data: dict) -> Release:
    tag = data["tag_name"]
    version = tag.lstrip("v")
    wanted = _platform_tag()
    asset = None
    sums = None
    for a in data.get("assets", []):
        name = a.get("name", "")
        if name == "SHA256SUMS.txt":
            sums = a["browser_download_url"]
        elif wanted and wanted in name and name.endswith((".tar.gz", ".zip")):
            asset = a
    return Release(
        tag=tag, version=version,
        asset_url=asset["browser_download_url"] if asset else None,
        asset_name=asset["name"] if asset else None,
        sums_url=sums, source_url=data.get("tarball_url"),
        prerelease=bool(data.get("prerelease") or data.get("draft")),
    )


class EngineInstaller:
    def __init__(self, settings: Settings, client: httpx.Client | None = None, api: str = RELEASES_API) -> None:
        self.settings = settings
        self.engines = settings.engines_dir
        self._api = api
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "coliollama"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        self._client = client or httpx.Client(headers=headers, follow_redirects=True, timeout=60.0)

    # -- discovery
    def current_dir(self) -> Path | None:
        return gpu.coli_dir(self.settings.resolve_coli())

    def current_version(self) -> str | None:
        return gpu.read_version(self.current_dir())

    def current_is_gpu_build(self) -> bool:
        directory = self.current_dir()
        return bool(directory and gpu.is_gpu_binary(directory / "colibri"))

    def gpu_build_possible(self) -> bool:
        return (
            sys.platform.startswith("linux")
            and self.settings.gpu.lower() not in ("none", "cpu", "off")
            and bool(gpu.nvidia_gpus())
            and gpu.find_nvcc() is not None
        )

    def _get_json(self, url: str):
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UpdateError(f"cannot reach GitHub releases: {exc}") from exc

    def latest_release(self) -> Release:
        for data in self._get_json(self._api):
            if not data.get("prerelease") and not data.get("draft"):
                return release_from_json(data)
        raise UpdateError("no published Colibrí release found")

    def release_for(self, version: str) -> Release:
        return release_from_json(self._get_json(f"{self._api}/tags/v{version}"))

    def check(self) -> tuple[UpdateStatus, Release]:
        latest = self.latest_release()
        current = self.current_version()
        newer = current is None or parse_version(latest.version) > parse_version(current)
        gpu_needed = self.gpu_build_possible() and not self.current_is_gpu_build()
        _write_state(self.engines, last_check=time.time())
        return UpdateStatus(
            current=current, latest=latest.version, update_available=newer,
            gpu_build_needed=gpu_needed,
            managed=managed_root(self.engines) is not None,
            pinned=self.settings.engine_pinned,
        ), latest

    # -- actions
    def update(self, say: Say = log.info, force: bool = False, build_gpu: bool | None = None,
               retry_gpu: bool = False) -> str | None:
        """Install the latest release if newer (or rebuild for GPU). Returns the new version or None."""
        with _install_lock(self.engines):
            status, latest = self.check()
            want_gpu = (status.gpu_build_needed or (force and self.gpu_build_possible())) if build_gpu is None else (
                build_gpu and self.gpu_build_possible())
            if build_gpu and not want_gpu:
                say("GPU build requested but NVIDIA GPU or nvcc (CUDA toolkit) was not found; skipping")
            if want_gpu and not status.update_available and not force and not retry_gpu:
                if status.current in _read_state(self.engines).get("gpu_build_failed", []):
                    want_gpu = False
            if status.update_available or force:
                target = latest
            elif want_gpu and status.current:
                try:
                    target = self.release_for(status.current)
                except UpdateError as exc:
                    say(f"cannot rebuild {status.current} for GPU: {exc}")
                    return None
            else:
                say(f"Colibrí {status.current} is up to date")
                return None
            return self._install(target, want_gpu, say)

    def _install(self, release: Release, build_gpu: bool, say: Say) -> str:
        if not release.asset_url:
            raise UpdateError(f"release {release.tag} has no asset for this platform ({_platform_tag()})")
        versions = self.engines / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.engines))
        try:
            archive = work / (release.asset_name or "engine.tar.gz")
            say(f"Downloading Colibrí {release.version} ({release.asset_name})")
            self._download(release.asset_url, archive, say)
            self._verify(archive, release)
            unpacked = work / "unpacked"
            unpacked.mkdir()
            safe_extract(archive, unpacked)
            root = _flatten_root(unpacked)
            found = gpu.read_version(root)
            if found and parse_version(found) != parse_version(release.version):
                raise UpdateError(f"archive version {found} does not match release {release.version}")
            rc = subprocess.run(
                [sys.executable, str(root / "coli"), "--help"],
                capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
            ).returncode
            if rc != 0:
                raise UpdateError("downloaded `coli` failed its self-check")
            built = False
            if build_gpu:
                built = self._build_gpu(release, root, work, say)
                if not built:
                    state = _read_state(self.engines)
                    failed = set(state.get("gpu_build_failed", [])) | {release.version}
                    _write_state(self.engines, gpu_build_failed=sorted(failed))
            final = versions / release.version
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(root), str(final))
            os.chmod(final / "coli", 0o755)
            self._activate(release.version, final)
            say(f"Colibrí {release.version} is now the main engine ({'GPU' if built else 'CPU'} build): {final}")
            return release.version
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _activate(self, version: str, path: Path) -> None:
        old = read_current(self.engines) or {}
        previous = old.get("version") if old.get("version") != version else old.get("previous")
        _write_json_atomic(
            _current_file(self.engines),
            {"version": version, "path": str(path), "previous": previous},
        )
        keep = {version, previous}
        for entry in (self.engines / "versions").iterdir():
            if entry.name not in keep:
                shutil.rmtree(entry, ignore_errors=True)

    def _download(self, url: str, dest: Path, say: Say) -> None:
        try:
            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                step = 0
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 16):
                        fh.write(chunk)
                        done += len(chunk)
                        if total and done * 4 // total > step:
                            step = done * 4 // total
                            say(f"  {done * 100 // total}% ({done // 1024} KiB / {total // 1024} KiB)")
        except httpx.HTTPError as exc:
            raise UpdateError(f"download failed: {exc}") from exc

    def _verify(self, archive: Path, release: Release) -> None:
        if not release.sums_url:
            raise UpdateError("release has no SHA256SUMS.txt; refusing to install unverified binaries")
        try:
            resp = self._client.get(release.sums_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpdateError(f"cannot fetch checksums: {exc}") from exc
        expected = None
        for line in resp.text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("*") == release.asset_name:
                expected = parts[0].lower()
        if not expected:
            raise UpdateError(f"no checksum listed for {release.asset_name}")
        digest = hashlib.sha256()
        with open(archive, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise UpdateError(f"checksum mismatch for {release.asset_name}")

    # -- CUDA build
    def _gpu_targets(self, root: Path) -> list[tuple[str, str]]:
        snippet = (
            "import json, family_registry as f\n"
            "out = []\n"
            "for fam in f.all_families():\n"
            "    if getattr(fam, 'supports_accelerator', False) and fam.id != 'deepseek_v4':\n"
            "        out.append([fam.build_target, getattr(fam, 'engine_artifact', fam.build_target)])\n"
            "print(json.dumps(out))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", snippet], cwd=root, capture_output=True, text=True, timeout=60,
                env={**os.environ, "PYTHONPATH": str(root)},
            )
            targets = [tuple(t) for t in json.loads(proc.stdout)]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            targets = []
        seen: dict[str, str] = {}
        for target, artifact in targets:
            seen.setdefault(artifact, target)
        return [(t, a) for a, t in seen.items()] or [("colibri", "colibri")]

    def _build_gpu(self, release: Release, root: Path, work: Path, say: Say) -> bool:
        nvcc = gpu.find_nvcc()
        if not (release.source_url and nvcc and shutil.which("make")):
            say("GPU build skipped: need nvcc, make and the release source")
            return False
        say("Building CUDA engines from source (this takes a few minutes)")
        src_archive = work / "source.tar.gz"
        self._download(release.source_url, src_archive, say)
        src = work / "source"
        src.mkdir()
        safe_extract(src_archive, src)
        top = next(p for p in src.iterdir() if p.is_dir())
        c_dir = top / "c"
        if not (c_dir / "Makefile").is_file():
            say("GPU build skipped: source has no c/Makefile")
            return False
        jobs = str(min(8, os.cpu_count() or 2))
        any_built = False
        for target, artifact in self._gpu_targets(root):
            say(f"  building {target} with CUDA")
            proc = subprocess.run(
                ["make", "-C", str(c_dir), target, "CUDA=1", f"CUDA_HOME={gpu.cuda_home(nvcc)}", "-j", jobs],
                capture_output=True, text=True, stdin=subprocess.DEVNULL,
            )
            built = c_dir / artifact
            if proc.returncode == 0 and gpu.is_gpu_binary(built):
                shutil.copy2(built, root / artifact)
                any_built = True
            elif proc.returncode == 0:
                say(f"  {target}: this engine has no CUDA code path, keeping the CPU binary")
            else:
                say(f"  {target}: CUDA build failed, keeping the CPU binary ({proc.stderr[-200:].strip()})")
        return any_built

    # -- background support
    def due(self) -> bool:
        last = _read_state(self.engines).get("last_check", 0)
        return time.time() - last >= self.settings.update_interval
