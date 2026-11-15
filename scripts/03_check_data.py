"""Data check: Kirchhoff residual per case and the three trend plots."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.cards import parse_pm  # noqa: E402
from leakml.devices import run_device  # noqa: E402
from leakml.plots import COLORS, style  # noqa: E402

plt = style()
rep = spec.ROOT / "results" / "partA"
rep.mkdir(parents=True, exist_ok=True)
cards = parse_pm()
nom = {k: {p: c.params[p] for p in spec.PROCESS_PARAMS} for k, c in cards.items()}

lines = ["# Data check (Part A)", "", "## Kirchhoff residual", "",
         "Residual = `|id+ig+is+ib|` per row (10,000 rows per case). A row passes when "
         "`residual <= 1e-6 * max(|id|,|ig|,|is|,|ib|) + 2e-15 A`: the relative term covers normal rows, "
         "the 2e-15 A term is the simulator's absolute noise floor (BSIM4 body currents sit at about 1e-14 A), "
         "which matters only where every terminal current is tiny, e.g. Vds close to 0.", "",
         "| Case | rows | max abs residual (A) | median abs residual (A) | max rel. residual | rows failing the rule |",
         "|---|---|---|---|---|---|"]
for case in spec.CASES:
    df = pd.read_csv(spec.ROOT / "data" / "partA" / f"{case}.csv")
    cur = df[list(spec.CURRENTS)].to_numpy()
    res, big = np.abs(cur.sum(1)), np.abs(cur).max(1)
    fail = (res > 1e-6 * big + 2e-15).sum()
    lines.append(f"| {case} | {len(df)} | {res.max():.2e} | {np.median(res):.2e} | {(res / big).max():.2e} | {fail} |")
    assert fail == 0, f"{case}: {fail} rows break the Kirchhoff rule"

def bias(case, **over):
    kind, vgm, vbm = spec.CASES[case]
    vdd = 1.0
    v = dict(vd=vdd if kind == "nmos" else 0.0, vs=0.0 if kind == "nmos" else vdd,
             vg=vgm * vdd, vb=vbm * vdd, temp=27.0, w=32e-9, **nom[kind])
    v.update(over)
    return kind, v


def sweep(case, key, values, term):
    out = []
    for x in values:
        kind, v = bias(case, **{key: x})
        out.append(abs(run_device(kind, v)[term]))
    return np.array(out)


def trend(name, key, values, cases, term, xlabel, title, scale=1.0):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for case in cases:
        ax.semilogy(np.asarray(values) * scale, sweep(case, key, values, term), "o-", ms=3,
                    color=COLORS[case], label=case)
    ax.set(xlabel=xlabel, ylabel=f"|{term}| (A)", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep / name)
    plt.close(fig)


temps = np.linspace(-40, 125, 12)
trend("trend_off_vs_temp.png", "temp", temps, ["NMOS_OFF", "PMOS_OFF"], "id",
      "temperature (°C)", "OFF current vs temperature")
trend_rows = []
for case in ("NMOS_OFF", "PMOS_OFF"):
    cur = sweep(case, "temp", temps, "id")
    trend_rows.append(f"- {case}: |id| {cur[0]:.2e} A at −40 °C, {cur[-1]:.2e} A at 125 °C (×{cur[-1] / cur[0]:.0f}).")

scales = np.linspace(0.9, 1.1, 9)
for name, key, cases, term, xl, title in [
        ("trend_off_vs_vth0.png", "vth0", ["NMOS_OFF", "PMOS_OFF"], "id", "vth0 / nominal", "OFF current vs vth0"),
        ("trend_gate_vs_toxe.png", "toxe", ["NMOS_ON", "PMOS_ON"], "ig", "toxe / nominal", "gate current vs toxe")]:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for case in cases:
        kind = spec.CASES[case][0]
        vals = nom[kind][key] * scales
        cur = sweep(case, key, vals, term)
        ax.semilogy(scales, cur, "o-", ms=3, color=COLORS[case], label=case)
        trend_rows.append(f"- {case}: |{term}| changes ×{cur[-1] / cur[0]:.2g} from {key} −10 % to +10 %.")
    ax.set(xlabel=xl, ylabel=f"|{term}| (A)", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep / name)
    plt.close(fig)

lines += ["", "## Trends (one parameter swept, others nominal, VDD = 1 V, W = 32 nm)", "",
          "- `trend_off_vs_temp.png`, `trend_off_vs_vth0.png`, `trend_gate_vs_toxe.png`", *trend_rows]
(rep / "DATA_CHECK.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
