"""Error metrics. Relative error uses a floor on the reference so that exact zeros
(e.g. Vds = 0) give a finite number instead of inf/NaN."""
import numpy as np


def rel_error(pred, true, floor: float = 1e-18):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return np.abs(pred - true) / np.maximum(np.abs(true), floor)


def r2(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bias(pred, true, floor: float = 1e-18) -> float:
    """Median signed relative error."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.median((pred - true) / np.maximum(np.abs(true), floor)))


def summarise(err) -> dict:
    err = np.asarray(err, float)
    mean = float(np.mean(err))
    return {"median": float(np.median(err)), "p95": float(np.quantile(err, 0.95)),
            "mean": mean, "accuracy": 1 - mean,
            "within_1pct": float(np.mean(err <= 0.01)), "max": float(np.max(err))}
