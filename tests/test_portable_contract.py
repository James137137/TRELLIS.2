from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from portable_env import APP_ROOT, PORTABLE_ROOT, configure_portable_environment


class PortableContractTests(unittest.TestCase):
    def test_all_declared_paths_stay_under_app_root(self) -> None:
        paths = configure_portable_environment(create=False)
        for name, path in paths.items():
            with self.subTest(name=name):
                path.resolve(strict=False).relative_to(APP_ROOT)

    def test_cache_environment_stays_portable(self) -> None:
        configure_portable_environment(create=False)
        keys = (
            "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN_PATH", "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
            "GRADIO_TEMP_DIR", "PIP_CACHE_DIR", "XDG_CACHE_HOME",
            "MPLCONFIGDIR", "TEMP", "TMP",
        )
        for key in keys:
            with self.subTest(key=key):
                Path(os.environ[key]).resolve(strict=False).relative_to(APP_ROOT)

    def test_manifest_has_valid_sha256_values(self) -> None:
        manifest_path = APP_ROOT / "windows" / "portable-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = [manifest["python"]["sha256"], manifest["python_dev"]["sha256"], manifest["pip"]["sha256"]]
        hashes.extend(item["sha256"] for item in manifest["windows_wheels"]["items"])
        for value in hashes:
            with self.subTest(value=value):
                self.assertEqual(len(value), 64)
                int(value, 16)

    def test_private_runtime_is_gitignored(self) -> None:
        gitignore = (APP_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".portable/", gitignore)
        self.assertEqual(PORTABLE_ROOT.parent, APP_ROOT)


if __name__ == "__main__":
    unittest.main()
