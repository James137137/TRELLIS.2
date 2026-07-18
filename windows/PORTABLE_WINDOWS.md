# TRELLIS.2 portable Windows application

## What the launcher changes

Nothing is installed into Windows. Setup creates only `.portable` in the
TRELLIS.2 folder. It does not request administrator privileges or modify PATH,
the registry, Windows features, file associations, Start Menu, or desktop.

The existing NVIDIA display driver and an existing Edge or Chrome executable are
the only machine-level prerequisites. The application was validated on Windows
11 with an NVIDIA GeForce RTX 4090 (24 GB).

## Folder layout

| Path | Purpose |
| --- | --- |
| `.portable/runtime` | Private Python 3.11 and Python packages |
| `.portable/models` | TRELLIS.2, DINOv3, and BiRefNet weights |
| `.portable/cache` | Torch, Triton, CUDA, Gradio, pip, and library caches |
| `.portable/secrets` | Folder-local Hugging Face read token |
| `.portable/browser-profile` | Dedicated Edge/Chrome app profile |
| `.portable/outputs` | Generated GLB files |
| `.portable/temp` | Process temporary files |
| `.portable/logs` | Backend startup logs |
| `.portable/run` | Per-launch readiness and shutdown files |

`portable_env.py` and `windows/portable-common.ps1` set every supported cache and
temporary environment variable before any ML or web library is imported.

## Native dependency stack

- Python 3.11.9 embeddable runtime and development files from Python.org/NuGet
- PyTorch 2.7.0, torchvision 0.22.0, and xFormers 0.0.30 for CUDA 12.8
- Triton-Windows 3.3.1.post21
- Windows wheels for CuMesh, FlexGEMM, O-Voxel, nvdiffrast, nvdiffrec, and the
  custom rasterizer from `visualbruno/ComfyUI-Trellis2` commit
  `438fe4e2a15bd29620a1cedad0a87c5afad6f81d`

Download URLs and SHA-256 checksums are pinned in `portable-manifest.json`.

## Hugging Face access

TRELLIS.2 requires Meta's gated DINOv3 encoder. Setup opens the access page in
the folder-local browser profile and stores the pasted read token only at
`.portable/secrets/huggingface-token`. Setup downloads these repositories:

- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `microsoft/TRELLIS.2-4B`
- `ZhengPeng7/BiRefNet`

Rerun `Setup TRELLIS 2.bat` if a download is interrupted.

## Startup and shutdown

`Launch TRELLIS 2.bat` starts a hidden Python server bound only to
`127.0.0.1` on a randomly selected free port. It then opens a dedicated browser
app window. When that window exits, the launcher writes a shutdown file, waits
for Gradio to close, and force-stops only that backend process if graceful
shutdown exceeds 20 seconds.

## Troubleshooting

- Setup output remains visible if installation fails.
- Startup errors are shown in a Windows dialog.
- Backend logs are under `.portable/logs`.
- If 1024 resolution runs out of VRAM, close other GPU applications or use 512.
- 1536 is available as a stress option but is not guaranteed within 24 GB VRAM.
