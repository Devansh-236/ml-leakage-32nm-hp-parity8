"""Part C SPICE reference: the parity checker at 200 corners x 256 input vectors (51,200 runs)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.parity import PARITY8, vector_bits  # noqa: E402
from leakml.reference import gate_sum_leakage, parity_reference  # noqa: E402
from leakml.sampling import draw_corners  # noqa: E402


def main():
    out = spec.ROOT / "data" / "partC"
    out.mkdir(parents=True, exist_ok=True)
    corners = draw_corners(spec.N_CORNERS, spec.SEEDS["corners_report"])
    df = parity_reference(corners)
    df.to_csv(out / "parity_spice.csv", index=False)
    print(f"{len(df)} runs, median leakage {df.leak.median():.3e} A")
    # does the circuit leakage equal the sum of its gates' leakages? (checked on a subset)
    rng = np.random.default_rng(spec.SEEDS["corners_report"])
    diffs = []
    for cid in rng.choice(spec.N_CORNERS, 20, replace=False):
        corner = corners.iloc[cid].to_dict()
        for v in rng.choice(256, 5, replace=False):
            whole = df[(df.corner_id == cid) & (df.vector == v)].leak.iloc[0]
            diffs.append(abs(gate_sum_leakage(PARITY8, corner, vector_bits(int(v))) - whole) / whole)
    diffs = np.array(diffs)
    msg = f"circuit vs sum of gates (100 samples): median {np.median(diffs):.3%}, max {diffs.max():.3%}"
    print(msg)
    (out / "gate_sum_check.txt").write_text(msg + "\n")


if __name__ == "__main__":
    main()
