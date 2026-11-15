"""Loading the four chosen transistor models."""
from pathlib import Path

import joblib

from . import spec


def load_models(directory: Path = spec.ROOT / "trained_models") -> dict:
    return {case: joblib.load(Path(directory) / f"{case}.joblib") for case in spec.CASES}
