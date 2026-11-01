"""Command line: ``wh run``, ``wh prepare``, ``wh explore``, ``wh results``."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

APPS = Path(__file__).parent / "apps"
DASHBOARD = APPS / "dashboard.py"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wh", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="full pipeline")
    run.add_argument("--no-causal", action="store_true", help="skip the DoWhy step")
    sub.add_parser("prepare", help="clean + validate data only")
    explore = sub.add_parser("explore", help="launch the Streamlit dashboard")
    explore.add_argument("--port", type=int, default=8501)
    sub.add_parser("results", help="launch the Dash results dashboard (needs the ml extra)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "explore":
        cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD)]
        subprocess.run([*cmd, f"--server.port={args.port}"], check=True)
        return

    from .pipeline import Pipeline  # model code needs the ml extra; the dashboard does not

    if args.cmd == "run":
        Pipeline().run(causal=not args.no_causal)
    elif args.cmd == "prepare":
        Pipeline().prepare()
    else:
        from .apps.results import create_app

        create_app().run(host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
