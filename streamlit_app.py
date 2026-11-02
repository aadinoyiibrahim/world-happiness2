
import sys
from pathlib import Path

try:
    from world_happiness.apps.dashboard import main
except ModuleNotFoundError:  
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from world_happiness.apps.dashboard import main

main()
