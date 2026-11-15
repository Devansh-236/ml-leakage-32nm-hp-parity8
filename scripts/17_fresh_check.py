"""Fresh-corner confirmation: 200 new corners (a seed used nowhere else), with new SPICE references for
all 18 gate cases and for the parity checker, scored with the unchanged final models. Nothing here
chooses or tunes a model. Writes data/fresh/ and results/FRESH_CHECK.md. SPICE results are reused if present."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.evaluate import gate_predictions, parity_predictions, pooled  # noqa: E402
from leakml.modelio import load_models  # noqa: E402
from leakml.reference import gate_reference, parity_reference  # noqa: E402
from leakml.sampling import draw_corners  # noqa: E402


def line(label, s, n):
    return (f"| {label} | {n:,} | {100 * s['median']:.2f} % | {100 * s['p95']:.2f} % | {100 * s['max']:.1f} % | "
            f"{100 * s['within_1pct']:.1f} % | {100 * s['mean']:.2f} % | {100 * s['accuracy']:.2f} % |")


def main():
    d = spec.ROOT / "data" / "fresh"
    d.mkdir(parents=True, exist_ok=True)
    corners = draw_corners(spec.N_CORNERS, spec.SEEDS["corners_fresh"])
    corners.to_csv(d / "corners_fresh.csv", index=False)
    if not (d / "gates_spice_fresh.csv").exists():
        gate_reference(corners).to_csv(d / "gates_spice_fresh.csv", index=False)
    if not (d / "parity_spice_fresh.csv").exists():
        parity_reference(corners).to_csv(d / "parity_spice_fresh.csv", index=False)
    gsp = pd.read_csv(d / "gates_spice_fresh.csv", dtype={"pattern": str})
    psp = pd.read_csv(d / "parity_spice_fresh.csv")
    models = load_models()
    gates = pd.concat([gate_predictions(models, corners, gsp, v) for v in ("a", "b")], ignore_index=True)
    par = parity_predictions(models, corners, psp)
    gates.to_csv(d / "gates_ml_fresh.csv", index=False)
    par.to_csv(d / "parity_ml_fresh.csv", index=False)
    rows = {"gates (a)": (pooled(gates[gates.variant == "a"]), int((gates.variant == "a").sum())),
            "gates (b)": (pooled(gates[gates.variant == "b"]), int((gates.variant == "b").sum())),
            "parity checker": (pooled(par), len(par))}
    pd.DataFrame([{"set": k, "runs": n, **s} for k, (s, n) in rows.items()]).to_csv(d / "fresh_summary.csv", index=False)
    text = ["# Fresh-corner check", "",
            f"200 corners drawn with seed {spec.SEEDS['corners_fresh']}, used nowhere else in the project. New SPICE references "
            "for every gate case and for the parity checker (all 256 vectors); the final models are unchanged.", "",
            "| Set | runs | median | p95 | max | within 1 % | mean error | mean accuracy |", "|---|---|---|---|---|---|---|---|",
            *[line(k, s, n) for k, (s, n) in rows.items()]]
    (spec.ROOT / "results" / "FRESH_CHECK.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
