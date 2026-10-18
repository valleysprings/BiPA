import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # bipa root
sys.path.insert(0, str(BASE))
