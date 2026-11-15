"""Part B: gate leakage from the transistor models against SPICE (deliverables 6 and 7)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.evaluate import gate_predictions, gate_table, pooled  # noqa: E402
from leakml.modelio import load_models  # noqa: E402
from leakml.plots import style  # noqa: E402


def main(tag="report"):
    d = spec.ROOT / "data" / "partB"
    rep = spec.ROOT / "results" / "partB"
    rep.mkdir(parents=True, exist_ok=True)
    corners = pd.read_csv(d / f"corners_{tag}.csv")
    spice = pd.read_csv(d / f"gates_spice_{tag}.csv", dtype={"pattern": str})
    models = load_models()
    pred = pd.concat([gate_predictions(models, corners, spice, v) for v in ("a", "b")], ignore_index=True)
    pred.to_csv(d / ("gates_ml.csv" if tag == "report" else f"gates_ml_{tag}.csv"), index=False)
    if tag != "report":
        return
    table = gate_table(pred)
    table.to_csv(d / "gates_table.csv", index=False)
    a, b = table[table.variant == "a"].set_index(["gate", "pattern"]), table[table.variant == "b"].set_index(["gate", "pattern"])
    lines = ["# Part B: gate leakage, ML vs SPICE (200 corners per case)", "",
             "Leakage = VDD supply current + current drawn by inputs held at VDD. Error = |ML − SPICE| / |SPICE| per corner.",
             "Variant (a) uses the SPICE-recorded internal node voltages (PDF step 4); variant (b) solves the internal "
             "node with the transistor models (no SPICE node data) and puts the output at its logic rail.", "",
             "| Gate | Input | SPICE median (A) | ML median (A) | bias | median err | p95 err | max err | within 1 % | (b) median err | (b) p95 err |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (gate, pat), r in a.iterrows():
        rb = b.loc[(gate, pat)]
        lines.append(f"| {gate} | {pat} | {r.spice_median:.3e} | {r.ml_median:.3e} | {100 * r.bias:+.2f} % | "
                     f"{100 * r.median_err:.2f} % | {100 * r.p95_err:.2f} % | {100 * r.max_err:.1f} % | {100 * r.within_1pct:.1f} % | "
                     f"{100 * rb.median_err:.2f} % | {100 * rb.p95_err:.2f} % |")
    for v, label in (("a", "(a) SPICE nodes"), ("b", "(b) model-solved nodes")):
        s = pooled(pred[pred.variant == v])
        lines.append(f"\n**Pooled {label}, {len(pred[pred.variant == v])} runs:** median {100 * s['median']:.2f} %, "
                     f"p95 {100 * s['p95']:.2f} %, mean {100 * s['mean']:.2f} %, mean accuracy {100 * s['accuracy']:.2f} %.")
    (rep / "GATES.md").write_text("\n".join(lines) + "\n")
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, v in zip(axes, ("a", "b")):
        g = pred[pred.variant == v]
        for gate, gg in g.groupby("gate"):
            ax.loglog(gg.spice_leak, gg.ml_leak, ".", ms=3, label=gate, alpha=0.6)
        lo, hi = g.spice_leak.min(), g.spice_leak.max()
        ax.loglog([lo, hi], [lo, hi], color="0.4", lw=0.8)
        ax.set(xlabel="SPICE leakage (A)", ylabel="ML leakage (A)", title=f"variant ({v})")
    axes[0].legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(rep / "gates_parity.png")
    print("\n".join(lines[-2:]))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "report"
    main(tag)
