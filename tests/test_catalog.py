from coliollama.core.hardware import Gpu, Hardware
from coliollama.registry.catalog import CatalogModel, assess

GB = 1_000_000_000


def hw(ram=16, disk=100, gpu=True):
    return Hardware(8, ram * GB, ram * GB, disk * GB, "/", [Gpu("x", 6 * GB, 6 * GB)] if gpu else [])


def test_small_model_fits():
    v = assess(CatalogModel("a/b", "qwen36", 10, 2, True), hw(), True)
    assert (v.level, v.processor) == ("fast", "GPU")


def test_large_model_streams_and_gpu_needs_cuda_engine():
    v = assess(CatalogModel("a/b", "qwen36", 100, 2, True), hw(disk=1000), False)
    assert (v.level, v.processor) == ("streamed", "GPU*")


def test_no_disk_and_cpu_only_family():
    assert assess(CatalogModel("a/b", "qwen36", 100, 2, True), hw(disk=10), True).level == "no-disk"
    assert assess(CatalogModel("a/b", "olmoe", 7, 2, False), hw(), True).processor == "CPU"
