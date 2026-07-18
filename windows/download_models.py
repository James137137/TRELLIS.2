"""Download all inference weights into the folder-local Hugging Face cache."""

from __future__ import annotations

import os
from pathlib import Path

from portable_env import configure_portable_environment


PATHS = configure_portable_environment()


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    token_path = Path(os.environ["HF_TOKEN_PATH"])
    if not token_path.is_file():
        raise RuntimeError("The folder-local Hugging Face token is missing.")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("The folder-local Hugging Face token is empty.")

    api = HfApi(token=token)
    repositories = (
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "microsoft/TRELLIS.2-4B",
        "ZhengPeng7/BiRefNet",
    )
    for repo_id in repositories:
        print(f"Checking access to {repo_id}...", flush=True)
        api.model_info(repo_id=repo_id, token=token)
        print(f"Downloading {repo_id}...", flush=True)
        snapshot_download(repo_id=repo_id, token=token)


if __name__ == "__main__":
    main()
