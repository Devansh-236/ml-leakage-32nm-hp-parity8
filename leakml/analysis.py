"""Error-analysis helpers: binned errors and the composition of the gate leakage."""
from typing import Sequence

import numpy as np
import pandas as pd


def bin_errors(df: pd.DataFrame, col: str, bins: Sequence[float]) -> pd.DataFrame:
    """Median / p95 error and count per bin of `col` (empty bins give NaN and n = 0)."""
    cut = pd.cut(df[col], bins, right=False)
    rows = {}
    for iv in cut.cat.categories:
        e = df.loc[cut == iv, "err"]
        rows[f"[{iv.left:g}, {iv.right:g})"] = {"median": e.median() if len(e) else np.nan,
                                                "p95": e.quantile(0.95) if len(e) else np.nan, "n": len(e)}
    return pd.DataFrame(rows).T.astype({"n": int})


def leakage_components(gate: str, bits: Sequence[int], corner: dict) -> dict:
    """Exact (SPICE-device) split of one gate case's leakage into channel, gate and body current."""
    from .circuits import GATES, simulate_circuit
    from .compose import leakage_parts
    from .oracle import OracleModels
    c = GATES[gate]
    r = simulate_circuit(c, corner, bits)
    row = {"vdd": corner["vdd"], "0": 0.0, **{n: corner["vdd"] * b for n, b in zip(c.inputs, bits)},
           **{n: r[n] for n in c.internal + (c.output,)}}
    parts = leakage_parts(c, pd.DataFrame([row]), pd.DataFrame([corner]), OracleModels())
    out = {k: float(v[0]) for k, v in parts.items()}
    out["total"] = out["channel"] + out["gate"] + out["body"]
    return out


def region_errors(case: str, model, df: pd.DataFrame, floor: float = 1e-18) -> pd.DataFrame:
    """Rows of `df` inside the bias region where gates use an OFF device (see
    training.region_weights), with the relative error of the model's id in column `err`."""
    from .training import region_weights
    mask = np.asarray(region_weights(case, df, 2.0)) > 1
    d = df[mask].reset_index(drop=True).copy()
    y = d["id"].to_numpy()
    d["err"] = np.abs(np.asarray(model.predict(d))[:, 0] - y) / np.maximum(np.abs(y), floor)
    return d
