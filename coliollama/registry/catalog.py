"""Models Colibrí can run, with size estimates and a fit verdict for the local hardware."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from coliollama.core.hardware import Hardware

# Converted size: 4-bit experts plus scales and wider dense weights, relative to parameter count.
CONVERTED_BYTES_PER_PARAM = 0.56
GB = 1e9


@dataclass(frozen=True)
class CatalogModel:
    repo: str
    family: str
    params_b: float
    source_bytes_per_param: float
    gpu_capable: bool
    note: str = ""
    verified: bool = True

    @property
    def source_bytes(self) -> float:
        return self.params_b * GB * self.source_bytes_per_param

    @property
    def converted_bytes(self) -> float:
        return self.params_b * GB * CONVERTED_BYTES_PER_PARAM


# Parameter counts come from the Hugging Face safetensors metadata of each repo.
CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel("tiny-random/qwen3.5-moe", "qwen36", 0.005, 2, True, "random weights; pipeline smoke test only"),
    CatalogModel("allenai/OLMoE-1B-7B-0125-Instruct", "olmoe", 6.9, 2, False, "smallest real model; CPU only"),
    CatalogModel("allenai/OLMoE-1B-7B-0924-Instruct", "olmoe", 6.9, 2, False, "CPU only"),
    CatalogModel("Qwen/Qwen3.5-35B-A3B", "qwen36", 36.0, 2, True),
    CatalogModel("Qwen/Qwen3.5-122B-A10B", "qwen36", 125.1, 2, True),
    CatalogModel("Qwen/Qwen3.5-397B-A17B", "qwen36", 403.4, 2, True),
    CatalogModel("Qwen/Qwen3.8-Flash-Next-FP8", "qwen38", 180.0, 1, True),
    CatalogModel("zai-org/GLM-5.3-Flash", "glm53", 321.3, 2, False, "no CUDA code path in Colibrí"),
    CatalogModel("zai-org/GLM-5.2-FP8", "glm", 753.3, 1, True),
    CatalogModel("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek_v4", 304.2, 1, False, "needs a special CUDA build"),
    CatalogModel("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41", 763.2, 1, False),
)

# model_type values Colibrí accepts, used for online discovery.
SUPPORTED_MODEL_TYPES = {
    "olmoe": ("olmoe", False), "qwen3_5_moe": ("qwen36", True), "qwen4_exp": ("qwen38", True),
    "glm_moe_dsa": ("glm", True), "kimi_k3": ("kimi", True),
}
_UNSUPPORTED_NAME = re.compile(r"gguf|gptq|awq|nvfp4|int4|int8|mlx|bnb|exl2|autoround|quantized", re.I)


@dataclass
class Verdict:
    model: CatalogModel
    level: str  # "fast" | "streamed" | "no-disk"
    processor: str
    disk_needed: float
    detail: str


def assess(model: CatalogModel, hw: Hardware, has_cuda_engine: bool) -> Verdict:
    peak_disk = model.source_bytes + model.converted_bytes  # raw download plus converted copy
    # Colibrí streams experts from disk; resident memory is the dense part plus a cache.
    ram = hw.ram_total or hw.ram_available
    use_gpu = model.gpu_capable and bool(hw.gpus)
    processor = "GPU" if use_gpu and has_cuda_engine else ("GPU*" if use_gpu else "CPU")
    if hw.disk_free < model.converted_bytes:
        return Verdict(model, "no-disk", processor, peak_disk,
                       f"needs {model.converted_bytes / GB:.0f} GB free, only {hw.disk_free / GB:.0f} GB")
    if hw.disk_free < peak_disk:
        level, detail = "streamed", "disk is tight: peak usage is download plus converted copy"
    else:
        level, detail = "fast", ""
    if model.converted_bytes <= ram * 0.7:
        level_ram = "fast"
    else:
        level_ram = "streamed"
        detail = (detail + "; " if detail else "") + "experts stream from disk, speed depends on the SSD"
    order = {"fast": 0, "streamed": 1}
    final = level if order[level] >= order[level_ram] else level_ram
    return Verdict(model, final, processor, peak_disk, detail)


def discover_online(limit: int = 15, timeout: float = 20.0) -> list[CatalogModel]:
    """Popular Hugging Face repos whose model_type Colibrí supports (unverified, may need manual conversion)."""
    known = {m.repo for m in CATALOG}
    found: list[CatalogModel] = []
    for model_type, (family, gpu_capable) in SUPPORTED_MODEL_TYPES.items():
        try:
            resp = httpx.get(
                "https://huggingface.co/api/models",
                params={"filter": model_type, "sort": "downloads", "direction": -1, "limit": limit,
                        "expand[]": ["safetensors", "gated"]},
                timeout=timeout,
            )
            resp.raise_for_status()
            items = resp.json()
        except (httpx.HTTPError, ValueError):
            continue
        for item in items:
            repo = item.get("id", "")
            total = (item.get("safetensors") or {}).get("total")
            if not repo or repo in known or _UNSUPPORTED_NAME.search(repo) or not total:
                continue
            known.add(repo)
            fp8 = "fp8" in repo.lower()
            found.append(CatalogModel(
                repo, family, total / GB, 1 if fp8 else 2, gpu_capable,
                "gated: needs HF_TOKEN" if item.get("gated") else "", verified=False,
            ))
    return found
