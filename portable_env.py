"""Folder-local runtime configuration for the portable Windows application."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict


APP_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = APP_ROOT / ".portable"


def _inside_app(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(APP_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"Portable path escapes the application folder: {resolved}") from exc
    return resolved


def configure_portable_environment(create: bool = True) -> Dict[str, Path]:
    """Route all application-owned state beneath ``.portable``.

    This function must run before importing Torch, Transformers, Gradio, OpenCV,
    or Triton so those libraries see the folder-local cache locations.
    """

    paths = {
        "root": PORTABLE_ROOT,
        "runtime": PORTABLE_ROOT / "runtime",
        "downloads": PORTABLE_ROOT / "downloads",
        "cache": PORTABLE_ROOT / "cache",
        "hf": PORTABLE_ROOT / "models" / "huggingface",
        "torch": PORTABLE_ROOT / "cache" / "torch",
        "torch_extensions": PORTABLE_ROOT / "cache" / "torch_extensions",
        "triton": PORTABLE_ROOT / "cache" / "triton",
        "cuda": PORTABLE_ROOT / "cache" / "cuda",
        "gradio": PORTABLE_ROOT / "cache" / "gradio",
        "pip": PORTABLE_ROOT / "cache" / "pip",
        "xdg": PORTABLE_ROOT / "cache" / "xdg",
        "matplotlib": PORTABLE_ROOT / "cache" / "matplotlib",
        "temp": PORTABLE_ROOT / "temp",
        "sessions": PORTABLE_ROOT / "sessions",
        "outputs": PORTABLE_ROOT / "outputs",
        "logs": PORTABLE_ROOT / "logs",
        "run": PORTABLE_ROOT / "run",
        "secrets": PORTABLE_ROOT / "secrets",
        "browser": PORTABLE_ROOT / "browser-profile",
    }
    paths = {name: _inside_app(path) for name, path in paths.items()}

    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

    token_path = paths["secrets"] / "huggingface-token"
    env = {
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "ATTN_BACKEND": "xformers",
        "SPARSE_ATTN_BACKEND": "xformers",
        "SPARSE_CONV_BACKEND": "flex_gemm",
        "HF_HOME": str(paths["hf"]),
        "HF_HUB_CACHE": str(paths["hf"] / "hub"),
        "HF_TOKEN_PATH": str(token_path),
        "TORCH_HOME": str(paths["torch"]),
        "TORCH_EXTENSIONS_DIR": str(paths["torch_extensions"]),
        "TRITON_CACHE_DIR": str(paths["triton"]),
        "CUDA_CACHE_PATH": str(paths["cuda"]),
        "GRADIO_TEMP_DIR": str(paths["gradio"]),
        "PIP_CACHE_DIR": str(paths["pip"]),
        "XDG_CACHE_HOME": str(paths["xdg"]),
        "MPLCONFIGDIR": str(paths["matplotlib"]),
        "TEMP": str(paths["temp"]),
        "TMP": str(paths["temp"]),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "DO_NOT_TRACK": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
        "PIP_USER": "no",
    }
    for key, value in env.items():
        os.environ[key] = value

    # expandable_segments is useful on Linux but unsupported by the Windows
    # CUDA allocator and produces a warning there.
    if os.name != "nt":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Make packaged CUDA DLLs discoverable without changing the machine PATH.
    dll_candidates = [
        paths["runtime"] / "python",
        paths["runtime"] / "python" / "Lib" / "site-packages" / "torch" / "lib",
        paths["runtime"] / "python" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]
    if hasattr(os, "add_dll_directory"):
        for candidate in dll_candidates:
            if candidate.is_dir():
                os.add_dll_directory(str(candidate))

    return paths


def portable_environment_report() -> str:
    paths = configure_portable_environment(create=False)
    return json.dumps({key: str(value) for key, value in paths.items()}, indent=2)


if __name__ == "__main__":
    print(portable_environment_report())
