"""Hardware inventory used to judge which models are worth pulling."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Gpu:
    name: str
    vram_total: int
    vram_free: int


@dataclass
class Hardware:
    cpu_threads: int
    ram_total: int
    ram_available: int
    disk_free: int
    disk_path: str
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def vram_total(self) -> int:
        return sum(g.vram_total for g in self.gpus)

    def to_dict(self) -> dict:
        return asdict(self)


def _memory() -> tuple[int, int]:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemFree"])
    except (OSError, KeyError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return total, total // 2
        except (OSError, ValueError):
            pass
    return 0, 0


def _nvidia() -> list[Gpu]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.splitlines() if out.returncode == 0 else []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                gpus.append(Gpu(parts[0], int(float(parts[1])) << 20, int(float(parts[2])) << 20))
            except ValueError:
                continue
    return gpus


def detect(models_dir: Path) -> Hardware:
    probe = models_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    total, available = _memory()
    return Hardware(
        cpu_threads=os.cpu_count() or 1,
        ram_total=total,
        ram_available=available,
        disk_free=shutil.disk_usage(probe).free,
        disk_path=str(probe),
        gpus=_nvidia(),
    )
