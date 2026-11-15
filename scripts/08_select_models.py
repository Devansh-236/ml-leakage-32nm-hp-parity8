"""Part A: best family per case chosen on validation, scored once on test (deliverable 5)."""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.families import FAMILIES  # noqa: E402
from leakml.reporting import comparison_markdown, eval_rows, pred_vs_spice_plot  # noqa: E402
from leakml.training import load_splits, select_best  # noqa: E402


def describe(model) -> str:
    i = model.info
    kind = f"{i.get('family', '?')}, {len(model.members)} member(s), {i.get('rows', '?')} rows"
    return kind + (", per-width targets" if model.per_width else ", plain targets")


def final_main():
    """Score the ladder's final models once on the test split and write the chosen-model table."""
    res = spec.ROOT / "data" / "partA_results"
    rep = spec.ROOT / "results" / "partA"
    rung = (res / "final" / "RUNG.txt").read_text().strip() if (res / "final" / "RUNG.txt").exists() else "L0"
    rows, desc = [], {}
    for case in spec.CASES:
        model = joblib.load(spec.ROOT / "trained_models" / f"{case}.joblib")
        sp = load_splits(case)
        rows += eval_rows(case, "final", model, sp.test, "test")
        pred_vs_spice_plot(case, model, sp.test, rep / f"pred_vs_spice_{case}.png")
        desc[case] = describe(model)
    df = pd.DataFrame(rows)
    df.to_csv(res / "chosen.csv", index=False)
    val = pd.read_csv(res / "compare_val.csv")
    text = ["# Part A: transistor models", "", comparison_markdown(val, "Level 0: model comparison (validation set, 7,000 training rows)"), ""]
    if (res / "ladder_log.csv").exists():
        log = pd.read_csv(res / "ladder_log.csv")
        text += ["## Escalation ladder", "",
                 "Gate-level mean accuracy on the development corners (not the reported ones) decides each step.", "",
                 "| Rung | variants (per case) | mean-median val. error (per case) | dev acc (a) | dev acc (b) |", "|---|---|---|---|---|"]
        for r, g in log.groupby("rung", sort=False):
            text.append(f"| {r} | " + ", ".join(f"{c}: {v}" for c, v in zip(g.case, g.variant)) + " | " +
                        ", ".join(f"{100 * e:.2f} %" for e in g.val_mean_median_err) +
                        f" | {100 * g.dev_acc_a.iloc[0]:.2f} % | {100 * g.dev_acc_b.iloc[0]:.2f} % |")
        text.append("")
    if (res / "data_size.csv").exists():
        ds = pd.read_csv(res / "data_size.csv")
        text += ["### Validation error versus training rows (one MLP per size)", "",
                 "| Case | " + " | ".join(f"{int(r)} rows" for r in sorted(ds.rows.unique())) + " |", "|---|" + "---|" * ds.rows.nunique()]
        for case in spec.CASES:
            g = ds[ds.case == case].sort_values("rows")
            text.append(f"| {case} | " + " | ".join(f"{100 * e:.2f} %" for e in g.val_mean_median_err) + " |")
        text.append("")
    text += [f"## Chosen models (final rung: {rung}; test set scored once)", "",
             "| Case | model | " + " | ".join(f"{t}: median / p95 (%)" for t in spec.CURRENTS) + " |", "|---|---|" + "---|" * 4]
    for case in spec.CASES:
        g = df[df.case == case].set_index("terminal")
        text.append(f"| {case} | {desc[case]} | " + " | ".join(f"{100 * g.loc[t, 'median']:.3g} / {100 * g.loc[t, 'p95']:.3g}" for t in spec.CURRENTS) + " |")
    (rep / "MODELS.md").write_text("\n".join(text) + "\n")


def main(source="L0"):
    res = spec.ROOT / "data" / "partA_results"
    rep = spec.ROOT / "results" / "partA"
    out = spec.ROOT / "trained_models"
    out.mkdir(exist_ok=True)
    rows, chosen = [], {}
    for case in spec.CASES:
        sp = load_splits(case)
        cands = {f: joblib.load(res / source / f"{case}_{f}.joblib") for f in FAMILIES
                 if (res / source / f"{case}_{f}.joblib").exists()}
        best = select_best(cands, sp.val)          # validation data only
        chosen[case] = best
        rows += eval_rows(case, best, cands[best], sp.test, "test")
        pred_vs_spice_plot(case, cands[best], sp.test, rep / f"pred_vs_spice_{case}.png")
        joblib.dump(cands[best], out / f"{case}.joblib")
        print(case, "->", best)
    df = pd.DataFrame(rows)
    df.to_csv(res / "chosen.csv", index=False)
    val = pd.read_csv(res / "compare_val.csv")
    text = ["# Part A: transistor models", "", comparison_markdown(val, "Model comparison (validation set)"), "",
            "## Chosen models (best on validation; test set scored once)", "",
            "| Case | chosen | " + " | ".join(f"{t}: median / p95 (%)" for t in spec.CURRENTS) + " |", "|---|---|" + "---|" * 4]
    for case in spec.CASES:
        g = df[df.case == case].set_index("terminal")
        text.append(f"| {case} | {chosen[case]} | " +
                    " | ".join(f"{100 * g.loc[t, 'median']:.3g} / {100 * g.loc[t, 'p95']:.3g}" for t in spec.CURRENTS) + " |")
    (rep / "MODELS.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the ladder's final models instead of selecting among level-0 families")
    final_main() if ap.parse_args().final else main()
