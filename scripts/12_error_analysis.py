"""Error analysis (report section) and the composition of the gate leakage at 32nm HP."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.analysis import bin_errors, leakage_components  # noqa: E402
from leakml.circuits import GATES  # noqa: E402
from leakml.plots import style  # noqa: E402


def fmt(t):
    return "; ".join(f"{i}: {100 * r['median']:.2f} % (p95 {100 * r['p95']:.1f} %, n={int(r['n'])})"
                     for i, r in t.iterrows() if r["n"])


def main():
    d = spec.ROOT / "data"
    rep = spec.ROOT / "results"
    corners = pd.read_csv(d / "partB" / "corners_report.csv")
    gates = pd.read_csv(d / "partB" / "gates_ml.csv", dtype={"pattern": str}).merge(corners, on="corner_id")
    par = pd.read_csv(d / "partC" / "parity_ml.csv").merge(corners, on="corner_id")
    dims = [("temp", "temperature (°C)", np.linspace(-40, 126, 6)),
            ("n_vth0", "NMOS vth0 (V)", np.linspace(0.4445, 0.5434, 5)),
            ("p_vth0", "PMOS |vth0|", None),
            ("wn", "Wn (m)", np.linspace(32e-9, 257e-9, 5)),
            ("wp", "Wp (m)", np.linspace(32e-9, 257e-9, 5)),
            ("vdd", "VDD (V)", np.linspace(0.9, 1.1001, 5))]
    lines = ["# Error analysis", "", "Error = |ML − SPICE| / |SPICE| per run. Medians and 95th percentiles per bin.", ""]
    plt = style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (col, label, bins) in zip(axes.ravel(), dims):
        if bins is None:
            gates["p_vth0_abs"] = gates.p_vth0.abs()
            par["p_vth0_abs"] = par.p_vth0.abs()
            bins = np.linspace(gates.p_vth0_abs.min(), gates.p_vth0_abs.max() + 1e-9, 6)
            c = "p_vth0_abs"
        else:
            c = col
        lines += [f"## By {label}", ""]
        for name, df in (("gates (a)", gates[gates.variant == "a"]), ("gates (b)", gates[gates.variant == "b"]),
                         ("parity checker", par)):
            t = bin_errors(df, c, bins)
            lines.append(f"- {name}: {fmt(t)}")
            mid = 0.5 * (np.asarray(bins)[:-1] + np.asarray(bins)[1:])          # bin centres in real units
            scale = 1e9 if col in ("wn", "wp") else 1.0
            ax.plot(mid * scale, 100 * t["median"], "o-", ms=3, label=name)
        ax.set(xlabel=label.replace("(m)", "(nm)"), ylabel="median error (%)")
        lines.append("")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(rep / "error_vs_corner.png")
    plt.close(fig)

    lines += ["## Worst gate cases (variant a)", ""]
    a = gates[gates.variant == "a"].sort_values("err", ascending=False).head(20)
    lines += ["| gate | input | err | temp (°C) | n_vth0 | p_vth0 | Wn (nm) | Wp (nm) | VDD |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in a.iterrows():
        lines.append(f"| {r.gate} | {r.pattern} | {100 * r.err:.1f} % | {r.temp:.0f} | {r.n_vth0:.3f} | {r.p_vth0:.3f} | "
                     f"{r.wn * 1e9:.0f} | {r.wp * 1e9:.0f} | {r.vdd:.3f} |")
    lines += ["", "## Bias of the ML leakage", ""]
    for name, df in (("gates (a)", gates[gates.variant == "a"]), ("parity checker", par)):
        signed = (df.ml_leak - df.spice_leak) / df.spice_leak
        lines.append(f"- {name}: median signed error {100 * signed.median():+.2f} %, "
                     f"{100 * (signed < 0).mean():.0f} % of runs under-predict")
    (rep / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n")

    # composition of the gate leakage (exact device currents from SPICE, report corners)
    sub = corners.iloc[:40].to_dict("records")
    rows = []
    for gate, c in GATES.items():
        for bits in itertools.product((0, 1), repeat=len(c.inputs)):
            comp = [leakage_components(gate, bits, corner) for corner in sub]
            tot = np.array([x["total"] for x in comp])
            share = {k: np.median([x[k] / x["total"] for x in comp]) for k in ("channel", "gate", "body")}
            rows.append({"gate": gate, "input": "".join(map(str, bits)), "median total (A)": np.median(tot), **{f"{k} share": v for k, v in share.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(d / "partB" / "leakage_components.csv", index=False)
    out = ["# What the gate leakage consists of (32nm HP)", "",
           "Exact split of each gate case's leakage (SPICE device currents, first 40 reported corners): "
           "channel = drain/source current of the OFF devices at the cut (sub-threshold), gate = gate tunnelling "
           "(including the current drawn by inputs held at VDD), body = junction/body current. Shares are medians of "
           "component / total.", "", "| Gate | Input | median total (A) | channel | gate | body |", "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        out.append(f"| {r.gate} | {r.input} | {r['median total (A)']:.3e} | {100 * r['channel share']:.1f} % | "
                   f"{100 * r['gate share']:.1f} % | {100 * r['body share']:.2f} % |")
    (rep / "partB" / "LEAKAGE_COMPONENTS.md").write_text("\n".join(out) + "\n")
    print("wrote ERROR_ANALYSIS.md and LEAKAGE_COMPONENTS.md")


if __name__ == "__main__":
    main()
