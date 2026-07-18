"""Fast native-Windows runtime and portability verification."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from portable_env import APP_ROOT, configure_portable_environment


PATHS = configure_portable_environment()


def _assert_inside(path: str | os.PathLike[str], label: str) -> None:
    resolved = Path(path).resolve(strict=False)
    try:
        resolved.relative_to(APP_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the portable folder: {resolved}") from exc


def main() -> None:
    import torch
    import xformers.ops as xops

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the NVIDIA GPU.")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 23 * 1024**3:
        raise RuntimeError(f"At least 23 GiB of VRAM is required; found {properties.total_memory / 1024**3:.1f} GiB.")

    modules = (
        "torch",
        "xformers",
        "triton",
        "cumesh",
        "flex_gemm",
        "o_voxel",
        "nvdiffrast",
        "nvdiffrec_render",
        "custom_rasterizer_kernel",
        "gradio",
        "transformers",
    )
    for name in modules:
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            _assert_inside(module_file, name)

    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN_PATH",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "GRADIO_TEMP_DIR",
        "PIP_CACHE_DIR",
        "TEMP",
        "TMP",
    ):
        _assert_inside(os.environ[key], key)

    q = torch.randn(1, 16, 1, 64, device="cuda", dtype=torch.float16)
    result = xops.memory_efficient_attention(q, q, q)
    if result.shape != q.shape:
        raise RuntimeError("xFormers attention returned an unexpected result.")
    del q, result
    torch.cuda.empty_cache()

    # Import the inference pipeline after all compiled dependencies are loaded.
    from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401

    print(f"Native runtime verified: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB VRAM")


if __name__ == "__main__":
    main()
