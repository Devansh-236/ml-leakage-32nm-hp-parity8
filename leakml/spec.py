"""Assignment, sampling ranges and seeds (DVD_ML_PROJECT.pdf sections 2-4).

Team assignment: models/32nm_HP.pm, nominal VDD 1.0 V, 8-bit parity checker.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "32nm_HP.pm"

VDD_NOM = 1.0                       # V, from the team sheet
L = 32e-9                           # m, feature size
W_RANGE = (1 * L, 8 * L)            # m
VDD_RANGE = (0.9 * VDD_NOM, 1.1 * VDD_NOM)
TEMP_RANGE = (-40.0, 125.0)         # degC
PROCESS_PARAMS = ("vth0", "toxe", "ndep", "u0", "xj")
PROCESS_SPREAD = 0.10

# case -> (device kind, vg as a multiple of VDD, vb as a multiple of VDD)
CASES = {
    "NMOS_ON": ("nmos", 1, 0),
    "NMOS_OFF": ("nmos", 0, 0),
    "PMOS_ON": ("pmos", 0, 1),
    "PMOS_OFF": ("pmos", 1, 1),
}

OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

CURRENTS = ("id", "ig", "is", "ib")
INPUT_COLS = ("vdd", "temp", "w", "vd", "vs", "vg", "vb") + PROCESS_PARAMS
DATA_COLS = INPUT_COLS + CURRENTS

SPLIT = (7000, 1500, 1500)
N_CORNERS = 200

SEEDS = {
    "split": 3201,
    "data": 3202,
    "extra": 3300,        # block k uses extra + k
    "corners_report": 3203,
    "corners_dev": 3204,
    "models": 3205,
    "diagnostic": 3206,   # evaluation-only rows at gate-relevant biases (scripts/14)
    "corners_fresh": 3207,  # 200 corners never used before, to confirm the final models (scripts/17)
}
