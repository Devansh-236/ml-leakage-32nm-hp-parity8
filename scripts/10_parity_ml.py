"""Part C: parity-checker leakage from the gate models against SPICE (deliverable 8)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.evaluate import parity_predictions, parity_table  # noqa: E402
from leakml.modelio import load_models  # noqa: E402
from leakml.plots import style  # noqa: E402


def main():
    d = spec.ROOT / "data"
    rep = spec.ROOT / "results" / "partC"
    rep.mkdir(parents=True, exist_ok=True)
    corners = pd.read_csv(d / "partB" / "corners_report.csv")
    spice = pd.read_csv(d / "partC" / "parity_spice.csv")
    pred = parity_predictions(load_models(), corners, spice)
    pred.to_csv(d / "partC" / "parity_ml.csv", index=False)
    table = parity_table(pred)
    table.to_csv(d / "partC" / "parity_table.csv")
    lines = ["# Part C: 8-bit parity checker, ML vs SPICE", "",
             f"{len(pred):,} runs (200 corners x 256 input vectors). ML leakage = sum of the 21 gate-model leakages "
             "(OR2, NAND2, AND2 per XOR2; gate inputs at their logic values, loading ignored; internal nodes solved "
             "by the transistor models, no circuit-level SPICE data).", "",
             "| Ones in input | runs | SPICE median (A) | ML median (A) | bias | median err | p95 err | max err | within 1 % | mean err |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in table.iterrows():
        lines.append(f"| {k} | {int(r.n)} | {r.spice_median:.3e} | {r.ml_median:.3e} | {100 * r.bias:+.2f} % | "
                     f"{100 * r.median_err:.2f} % | {100 * r.p95_err:.2f} % | {100 * r.max_err:.1f} % | "
                     f"{100 * r.within_1pct:.1f} % | {100 * r.mean_err:.2f} % |")
    a = table.loc["all"]
    lines.append(f"\n**Overall mean accuracy: {100 * (1 - a.mean_err):.2f} %** (mean error {100 * a.mean_err:.2f} %).")
    (rep / "PARITY.md").write_text("\n".join(lines) + "\n")
    plt = style()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.loglog(pred.spice_leak, pred.ml_leak, ".", ms=1.5, alpha=0.3, color="#1b6ca8")
    lo, hi = pred.spice_leak.min(), pred.spice_leak.max()
    ax.loglog([lo, hi], [lo, hi], color="0.4", lw=0.8)
    ax.set(xlabel="SPICE leakage (A)", ylabel="ML leakage (A)", title="8-bit parity checker")
    fig.tight_layout()
    fig.savefig(rep / "parity_parity.png")
    print(lines[-1])


if __name__ == "__main__":
    main()
