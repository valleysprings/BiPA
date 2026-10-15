import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE.parent), str(BASE.parent / "scripts/analysis")]
