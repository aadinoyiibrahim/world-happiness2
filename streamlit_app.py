"""Entry point for Streamlit Community Cloud (and ``streamlit run streamlit_app.py``)."""

import sys
from pathlib import Path

try:
    from world_happiness.apps.dashboard import main
except ModuleNotFoundError:  # the host installed the dependencies but not the package
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from world_happiness.apps.dashboard import main

main()
