"""Controlled localhost server used by the portable Windows launcher."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from portable_env import configure_portable_environment


configure_portable_environment()


def _portable_file(value: str) -> Path:
    from portable_env import APP_ROOT

    path = Path(value).resolve(strict=False)
    try:
        path.relative_to(APP_ROOT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Path must stay inside {APP_ROOT}") from exc
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--ready-file", required=True, type=_portable_file)
    parser.add_argument("--shutdown-file", required=True, type=_portable_file)
    parser.add_argument("--error-file", required=True, type=_portable_file)
    args = parser.parse_args()

    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid localhost port.")

    demo = None
    try:
        from app import css, demo, head, initialize_ui_assets

        initialize_ui_assets()
        launch_result = demo.launch(
            server_name=args.host,
            server_port=args.port,
            inbrowser=False,
            prevent_thread_lock=True,
            quiet=True,
            css=css,
            head=head,
        )
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(
            json.dumps({"url": f"http://{args.host}:{args.port}", "pid": os.getpid()}),
            encoding="utf-8",
        )
        while not args.shutdown_file.exists():
            time.sleep(0.25)
    except BaseException:
        args.error_file.parent.mkdir(parents=True, exist_ok=True)
        args.error_file.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if demo is not None:
            demo.close()


if __name__ == "__main__":
    main()
