"""Resolve a model reference to a usable registered model.

Downloads from the Hugging Face Hub when needed, then (unless disabled) verifies
that Colibrí can load the directory and converts it when it cannot.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path

from coliollama.core.config import Settings
from coliollama.registry.local_store import LocalStore, ModelEntry
from coliollama.registry.readiness import ModelChecker, ModelConverter, ReadinessError

# `organization/repository-name`; segments must start alphanumeric, so ".." is impossible.
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

Downloader = Callable[..., str]


class ModelResolutionError(Exception):
    pass


def looks_like_repo_id(ref: str) -> bool:
    return bool(_REPO_ID.match(ref))


def _snapshot_download(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


class HuggingFaceResolver:
    def __init__(
        self,
        store: LocalStore,
        settings: Settings,
        downloader: Downloader | None = None,
        checker: ModelChecker | None = None,
        converter: ModelConverter | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._downloader = downloader or _snapshot_download
        self._checker = checker or ModelChecker(settings)
        self._converter = converter or ModelConverter(settings)

    def local_dir_for(self, repo_id: str) -> Path:
        return self._settings.models_dir / repo_id.replace("/", "--")

    def converted_dir_for(self, repo_id: str) -> Path:
        return self._settings.models_dir / (repo_id.replace("/", "--") + "-coli")

    def ensure(
        self,
        ref: str,
        *,
        revision: str | None = None,
        allow_download: bool = True,
        convert: bool = True,
        keep_source: bool = False,
        on_status: Callable[[str], None] | None = None,
    ) -> ModelEntry:
        """Return a registered, ready model for `ref` (registry name, local directory or HF repo ID).

        With `convert=False` no readiness check or conversion is done.
        """
        say = on_status or (lambda _msg: None)

        entry = self._store.get(ref)
        if entry and entry.exists:
            if not convert or entry.verified:
                return entry
            return self._finalize(entry.name, Path(entry.path), entry.repo_id, True, keep_source, say)

        local = Path(ref).expanduser()
        if local.is_dir():
            return self._finalize(local.resolve().name, local.resolve(), None, convert, keep_source, say)

        if not looks_like_repo_id(ref):
            raise ModelResolutionError(
                f"model {ref!r} is not registered, is not a directory and is not a "
                "Hugging Face repo ID (organization/repository)"
            )
        if not allow_download:
            raise ModelResolutionError(f"model {ref!r} is not available locally; run `coliollama pull {ref}`")

        say(f"Downloading {ref} from Hugging Face ...")
        target = self.local_dir_for(ref)
        try:
            self._downloader(
                repo_id=ref, local_dir=str(target), revision=revision, token=self._settings.hf_token
            )
        except Exception as exc:
            raise ModelResolutionError(f"download of {ref!r} failed: {exc}") from exc
        if revision and convert:
            say(f"Note: `coli convert` has no revision option; a conversion would use the default branch, not {revision!r}")
        return self._finalize(ref, target, ref, convert, keep_source, say)

    def _finalize(
        self, name: str, path: Path, repo_id: str | None, convert: bool, keep_source: bool, say
    ) -> ModelEntry:
        if not convert:
            say(f"Registered {name} at {path} (readiness check skipped)")
            return self._store.add(name, path, repo_id=repo_id)

        say(f"Checking that Colibrí can use {path} ...")
        try:
            report = self._checker.check(path)
        except ReadinessError as exc:
            raise ModelResolutionError(str(exc)) from exc
        if report.usable:
            self._warn_environment(report, say)
            say(f"Registered {name} at {path}")
            return self._store.add(name, path, repo_id=repo_id, verified=True)

        problems = "; ".join(report.model_problems)
        if not repo_id:
            raise ModelResolutionError(
                f"{path} is not directly usable ({problems}). Automatic conversion only works for "
                "Hugging Face repo IDs; convert it manually with `coli convert`, or use --no-convert "
                "to register it as is"
            )

        say(f"Model is not directly usable ({problems}); converting with `coli convert` ...")
        out_dir = self.converted_dir_for(repo_id)
        try:
            self._converter.convert(repo_id, out_dir, source_dir=path)
            report = self._checker.check(out_dir)
        except ReadinessError as exc:
            raise ModelResolutionError(str(exc)) from exc
        if not report.usable:
            raise ModelResolutionError(
                f"converted model in {out_dir} is still not usable: {'; '.join(report.model_problems)}"
            )
        self._warn_environment(report, say)
        if not keep_source and self._is_managed(path):
            shutil.rmtree(path, ignore_errors=True)
            say(f"Removed raw download {path} (use --keep-source to keep it)")
        say(f"Registered {name} at {out_dir}")
        return self._store.add(name, out_dir, repo_id=repo_id, verified=True)

    def _is_managed(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self._settings.models_dir.resolve())
        except OSError:
            return False

    @staticmethod
    def _warn_environment(report, say) -> None:
        for problem in report.environment_problems:
            say(f"Warning (not a model problem): {problem}")
