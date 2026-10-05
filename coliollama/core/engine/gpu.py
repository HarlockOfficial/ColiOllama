"""GPU detection and the decision whether to start the engine on the GPU."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def nvidia_gpus() -> list[str]:
    """Names of NVIDIA GPUs visible to the driver (empty if none or no driver)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def find_nvcc() -> Path | None:
    candidates = [shutil.which("nvcc")]
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if home:
            candidates.append(str(Path(home) / "bin" / "nvcc"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def cuda_home(nvcc: Path) -> Path:
    return nvcc.parent.parent


def is_gpu_binary(binary: Path) -> bool:
    """A Colibrí engine is GPU-capable when it links the CUDA or HIP runtime."""
    ldd = shutil.which("ldd")
    if not ldd or not binary.is_file():
        return False
    try:
        out = subprocess.run([ldd, str(binary)], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(re.search(r"libcudart|libamdhip64", out))


_doctor_cache: dict[tuple, bool] = {}


def gpu_usable(command: tuple[str, ...], model_path: str) -> bool:
    """Ask `coli doctor` whether this engine build and this machine can run the model on GPU."""
    key = (command, model_path)
    if key not in _doctor_cache:
        usable = False
        try:
            proc = subprocess.run(
                [*command, "doctor", "--model", model_path, "--json"],
                capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            )
            checks = json.loads(proc.stdout).get("checks", [])
            usable = any(c.get("id") == "accelerator.gpu" and c.get("status") == "pass" for c in checks)
        except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
            pass
        _doctor_cache[key] = usable
    return _doctor_cache[key]


def engine_gpu_args(
    setting: str, command: tuple[str, ...], model_path: str, user_args: tuple[str, ...]
) -> tuple[list[str], str]:
    """Return (extra engine args, processor label) for the configured `gpu` setting."""
    if any(a == "--gpu" or a.startswith("--gpu=") for a in user_args):
        return [], "custom"
    setting = (setting or "auto").strip().lower()
    if setting in ("none", "cpu", "off"):
        return ["--gpu", "none"], "cpu"
    if setting == "auto":
        if gpu_usable(command, model_path):
            return ["--gpu", "auto", "--auto-tier"], "gpu"
        return [], "cpu"
    return ["--gpu", setting, "--auto-tier"], "gpu"


def coli_dir(command: tuple[str, ...] | None) -> Path | None:
    """Directory holding the `coli` script of a resolved launcher."""
    for part in reversed(command or ()):
        if Path(part).name == "coli" and Path(part).is_file():
            return Path(part).resolve().parent
    return None


def read_version(directory: Path | None) -> str | None:
    if directory is None:
        return None
    try:
        text = (directory / "version.py").read_text()
    except OSError:
        return None
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else None


