import sys

from src.core.layout import OPTIMIZERS

sys.path[:0] = [str(OPTIMIZERS / "skillopt"), str(OPTIMIZERS / "skilladam"), str(OPTIMIZERS / "gepa/src")]
