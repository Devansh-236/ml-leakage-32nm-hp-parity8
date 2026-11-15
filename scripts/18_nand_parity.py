"""Second XOR2 topology: the parity checker with every XOR2 built from four NAND2 (28 NAND2, 112
transistors), simulated as one netlist at the 200 reported corners for all 256 vectors and compared
with the gate-level ML estimate from the same, unchanged models. Writes data/partC/parity_nand_*.csv
and results/partC/PARITY_NAND.md. SPICE results are reused if present."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.evaluate import parity_predictions, parity_table  # noqa: E402
from leakml.modelio import load_models  # noqa: E402
from leakml.parity import PARITY8_NAND, vector_bits  # noqa: E402
from leakml.reference import gate_sum_leakage, parity_reference  # noqa: E402


def main():
    d = spec.ROOT / "data" / "partC"
    corners = pd.read_csv(spec.ROOT / "data" / "partB" / "corners_report.csv")
    f = d / "parity_nand_spice.csv"
    if not f.exists():
        parity_reference(corners, circuit=PARITY8_NAND).to_csv(f, index=False)
    spice = pd.read_csv(f)
    wrong = int(((spice.p > 0.5).astype(int) != spice.vector.map(lambda v: bin(int(v)).count("1") % 2)).sum())
    pred = parity_predictions(load_models(), corners, spice, circuit=PARITY8_NAND)
    pred.to_csv(d / "parity_nand_ml.csv", index=False)
    table = parity_table(pred)
    table.to_csv(d / "parity_nand_table.csv")
    rng = np.random.default_rng(spec.SEEDS["corners_report"])
    diffs = []
    for cid in rng.choice(spec.N_CORNERS, 20, replace=False):
        corner = corners.iloc[cid].to_dict()
        for v in rng.choice(256, 5, replace=False):
            whole = spice[(spice.corner_id == cid) & (spice.vector == v)].leak.iloc[0]
            diffs.append(abs(gate_sum_leakage(PARITY8_NAND, corner, vector_bits(int(v))) - whole) / whole)
    diffs = np.array(diffs)
    a = table.loc["all"]
    text = ["# Parity checker built from NAND2 only", "",
            f"{len(pred):,} runs (200 reported corners x 256 vectors), each XOR2 = 4 NAND2 (28 NAND2, 112 transistors). "
            f"Output logic wrong in {wrong} runs. Circuit vs sum of its gates in SPICE (100 samples): median "
            f"{100 * np.median(diffs):.3f} %, max {100 * diffs.max():.3f} %.", "",
            "| Ones in input | runs | SPICE median (A) | ML median (A) | bias | median err | p95 err | max err | within 1 % | mean err |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in table.iterrows():
        text.append(f"| {k} | {int(r.n)} | {r.spice_median:.3e} | {r.ml_median:.3e} | {100 * r.bias:+.2f} % | "
                    f"{100 * r.median_err:.2f} % | {100 * r.p95_err:.2f} % | {100 * r.max_err:.1f} % | "
                    f"{100 * r.within_1pct:.1f} % | {100 * r.mean_err:.2f} % |")
    text.append(f"\n**Overall mean accuracy: {100 * (1 - a.mean_err):.2f} %** (mean error {100 * a.mean_err:.2f} %).")
    (spec.ROOT / "results" / "partC" / "PARITY_NAND.md").write_text("\n".join(text) + "\n")
    (d / "gate_sum_check_nand.txt").write_text(f"circuit vs sum of gates (100 samples): median {np.median(diffs):.3%}, "
                                               f"max {diffs.max():.3%}\n")
    print("\n".join(text[:3] + text[-1:]))


if __name__ == "__main__":
    main()
