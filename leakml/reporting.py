"""Part A tables and plots shared by the comparison, selection and ladder scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import spec
from .plots import COLORS, style


def eval_rows(case, family, model, df, split) -> list:
    from .training import evaluate
    return [{"case": case, "family": family, "split": split, "terminal": t, **m}
            for t, m in evaluate(model, df).items()]


def comparison_markdown(df: pd.DataFrame, title: str) -> str:
    """4 families x 4 cases: R², median and p95 relative error per terminal."""
    lines = [f"## {title}", "",
             "R² is computed on the signed-log scale (currents span many decades; linear R² is in "
             "`compare_val.csv`). Errors are per-sample relative errors in %.", ""]
    for case in spec.CASES:
        lines += [f"### {case}", "", "| Family | " + " | ".join(f"{t}: R² / med / p95" for t in spec.CURRENTS) + " |",
                  "|---|" + "---|" * 4]
        for fam, g in df[df.case == case].groupby("family", sort=False):
            g = g.set_index("terminal")
            cells = [f"{g.loc[t, 'r2']:.5f} / {100 * g.loc[t, 'median']:.3g} / {100 * g.loc[t, 'p95']:.3g}"
                     for t in spec.CURRENTS]
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def pred_vs_spice_plot(case, model, test: pd.DataFrame, path: Path):
    plt = style()
    pred, true = model.predict(test), test[list(spec.CURRENTS)].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for j, (ax, term) in enumerate(zip(axes, spec.CURRENTS)):
        a, b = np.abs(true[:, j]) + 1e-20, np.abs(pred[:, j]) + 1e-20
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.loglog([lo, hi], [lo, hi], color="0.5", lw=0.8)
        ax.loglog(a, b, ".", ms=2.5, color=COLORS[case], alpha=0.5)
        ax.set(xlabel=f"|{term}| SPICE (A)", ylabel=f"|{term}| ML (A)", title=term)
    fig.suptitle(f"{case}: predicted vs SPICE, test set (1,500 samples)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
