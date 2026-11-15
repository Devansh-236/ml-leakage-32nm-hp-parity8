"""Diagnostic (evaluation only, never used for training): transistor-level error of the final OFF models
at the bias region gates use, binned by temperature, |vth0|, width and VDD. New SPICE rows are drawn with a
separate seed, concentrated near the rails, so that enough fall inside the region."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.analysis import bin_errors, region_errors  # noqa: E402
from leakml.devices import generate_case  # noqa: E402

ROWS = 8000
BINS = {"temp": [-40, 0, 40, 80, 100, 125.1], "w_nm": [32, 70, 110, 160, 210, 257], "vdd": [0.9, 0.95, 1.0, 1.05, 1.1001]}


def main():
    out = spec.ROOT / "data" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Transistor-level error in the gate-relevant bias region (OFF cases)", "",
             f"{ROWS} fresh SPICE rows per case (seed {spec.SEEDS['diagnostic']}, 80 % of Vd/Vs drawn next to the rails), never used for "
             "training or selection. Only rows inside the region gates use are kept (NMOS: Vs < 0.2 VDD, Vd > 0.7 VDD; PMOS mirrored). "
             "Error = |id_ML - id_SPICE| / |id_SPICE|; entries are median % (n).", ""]
    for case in ("NMOS_OFF", "PMOS_OFF"):
        f = out / f"{case}.csv"
        if f.exists():
            df = pd.read_csv(f)
        else:
            df, _ = generate_case(case, ROWS, spec.SEEDS["diagnostic"], edge_fraction=0.8)
            df.to_csv(f, index=False)
        d = region_errors(case, joblib.load(spec.ROOT / "trained_models" / f"{case}.joblib"), df)
        d["w_nm"] = d.w * 1e9
        d["abs_vth0"] = d.vth0.abs()
        q = np.quantile(d.abs_vth0, np.linspace(0, 1, 6))
        q[-1] += 1e-9
        lines += [f"## {case}: {len(d)} rows in the region, median error {100 * d.err.median():.2f} %, p95 {100 * d.err.quantile(0.95):.1f} %, "
                  f"median |id| {d.id.abs().median():.1e} A", ""]
        for name, col, edges in (("temperature (C)", "temp", BINS["temp"]), (f"|vth0| quintiles ({q[0]:.3f} to {q[-1]:.3f} V)", "abs_vth0", q),
                                 ("width (nm)", "w_nm", BINS["w_nm"]), ("VDD (V)", "vdd", BINS["vdd"])):
            t = bin_errors(d, col, edges)
            lines.append(f"- {name}: " + "; ".join(f"{k}: {100 * r['median']:.2f} % (n={int(r['n'])})" for k, r in t.iterrows()))
        big = d.id.abs() > d.id.abs().median()
        lines += [f"- split at the median |id|: above {100 * d.err[big].median():.2f} %, below {100 * d.err[~big].median():.2f} %", ""]
    (spec.ROOT / "results" / "partA" / "REGION_DIAGNOSTIC.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
