"""Local model registry: a JSON file mapping model names to absolute paths."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelEntry:
    name: str
    path: str
    repo_id: str | None = None
    added_at: float = 0.0
    verified: bool = False  # passed `coli doctor` (directly usable by the engine)

    @property
    def exists(self) -> bool:
        return Path(self.path).is_dir()

    def size_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.path):
            for fname in files:
                try:
                    total += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass
        return total


def normalize_name(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


class LocalStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        try:
            return json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}

    def _save(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self._path)

    def list(self) -> list[ModelEntry]:
        with self._lock:
            return [ModelEntry(**v) for _, v in sorted(self._load().items())]

    def get(self, name: str) -> ModelEntry | None:
        key = normalize_name(name)
        with self._lock:
            raw = self._load().get(key)
        return ModelEntry(**raw) if raw else None

    def add(
        self, name: str, path: str | Path, repo_id: str | None = None, verified: bool = False
    ) -> ModelEntry:
        entry = ModelEntry(
            name=normalize_name(name),
            path=str(Path(path).expanduser().resolve()),
            repo_id=repo_id,
            added_at=time.time(),
            verified=verified,
        )
        with self._lock:
            data = self._load()
            data[entry.name] = asdict(entry)
            self._save(data)
        return entry

    def remove(self, name: str) -> bool:
        key = normalize_name(name)
        with self._lock:
            data = self._load()
            if key not in data:
                return False
            del data[key]
            self._save(data)
        return True
