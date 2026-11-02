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
    exp = sub.add_parser(
        "export-map", help="save the animated map as a GIF (needs the export extra)"
    )
    exp.add_argument("--outdir", default="docs/assets/", help="folder for the GIF")
    exp.add_argument("--scope", default="World", help="World or a continent, e.g. Europe")
    exp.add_argument("--mode", choices=["timeline", "globe"], default="timeline")
    exp.add_argument("--theme", choices=["dark", "light"], default="dark")
    exp.add_argument("--n", type=int, default=5, help="how many top/worst countries to ring")
    exp.add_argument("--year", type=int, default=None, help="globe mode: year to show")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "explore":
        cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD)]
        subprocess.run([*cmd, f"--server.port={args.port}"], check=True)
        return

    if args.cmd == "export-map":
        from .config import Settings
        from .data import WHRDataset
        from .export import export_map

        df = WHRDataset.from_excel(Settings().raw_path).df
        outdir = args.outdir if args.outdir.endswith("/") else args.outdir + "/"
        path = export_map(df, outdir, args.scope, args.mode, args.theme, n=args.n, year=args.year)
        logging.getLogger("wh").info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return

    from .pipeline import Pipeline

    if args.cmd == "run":
        Pipeline().run(causal=not args.no_causal)
    elif args.cmd == "prepare":
        Pipeline().prepare()
    else:
        from .apps.results import create_app

        create_app().run(host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
