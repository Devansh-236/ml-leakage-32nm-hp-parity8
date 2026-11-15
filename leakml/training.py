"""Splits, case models, evaluation and model selection for Part A."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from . import spec
from .families import make_family
from .features import build_features, fit_floor, from_target, to_target, weff
from .metrics import r2, rel_error, summarise


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


EDGE_DIR = spec.ROOT / "data" / "partA_edge"


def load_splits(case: str, extra_blocks: int = 0, edge_blocks: int = 0) -> Splits:
    """The case's 10,000-row CSV split 7,000/1,500/1,500 (fixed seed). Extra blocks from
    data/partA_extra, and edge blocks from data/partA_edge (Vd, Vs concentrated at the rails), are
    appended to the training part only; validation and test stay the uniform PDF draws."""
    df = pd.read_csv(spec.ROOT / "data" / "partA" / f"{case}.csv")
    perm = np.random.default_rng(spec.SEEDS["split"]).permutation(len(df))
    a, b = spec.SPLIT[0], spec.SPLIT[0] + spec.SPLIT[1]
    train, val, test = (df.iloc[perm[s]].reset_index(drop=True) for s in (slice(0, a), slice(a, b), slice(b, None)))
    if extra_blocks:
        blocks = [pd.read_csv(spec.ROOT / "data" / "partA_extra" / f"{case}_{k}.csv") for k in range(1, extra_blocks + 1)]
        train = pd.concat([train, *blocks], ignore_index=True)
    if edge_blocks:
        blocks = [pd.read_csv(EDGE_DIR / f"{case}_{k}.csv") for k in range(1, edge_blocks + 1)]
        train = pd.concat([train, *blocks], ignore_index=True)
    return Splits(train, val, test)


class CaseModel:
    """Predicts the four terminal currents of one case. Members (one or more fitted
    families) are averaged in signed-log target space."""

    def __init__(self, case, kind, members, floors, info=None, per_width=False):
        self.case, self.kind, self.members, self.floors = case, kind, list(members), np.asarray(floors, float)
        self.info, self.per_width = info or {}, per_width

    def predict(self, df) -> np.ndarray:
        x = build_features(df, self.kind)
        t = np.mean([m.predict(x) for m in self.members], axis=0)
        cur = np.column_stack([from_target(t[:, j], self.floors[j]) for j in range(t.shape[1])])
        return cur * weff(df["w"].to_numpy())[:, None] if self.per_width else cur


def region_weights(case: str, df: pd.DataFrame, factor: float) -> np.ndarray:
    """Training weights that emphasise the bias region where gates use an OFF device (source at its
    supply rail, drain at the opposite rail). NMOS_OFF: vs < 0.2 VDD and vd > 0.7 VDD; PMOS_OFF:
    vs > 0.8 VDD and vd < 0.3 VDD. ON cases are not reweighted."""
    w = np.ones(len(df))
    if case == "NMOS_OFF":
        w[((df.vs < 0.2 * df.vdd) & (df.vd > 0.7 * df.vdd)).to_numpy()] = factor
    elif case == "PMOS_OFF":
        w[((df.vs > 0.8 * df.vdd) & (df.vd < 0.3 * df.vdd)).to_numpy()] = factor
    return w


def fit_case_model(case: str, family: str, splits: Splits, seed: int, per_width: bool = True,
                   region_weight: float = 1.0, **kw) -> CaseModel:
    """Fit one family on the training split. Targets are signed-log currents, per effective
    width when `per_width`. `region_weight` > 1 weights the gate-relevant bias region (OFF cases)."""
    kind = spec.CASES[case][0]
    cur = splits.train[list(spec.CURRENTS)].to_numpy()
    if per_width:
        cur = cur / weff(splits.train["w"].to_numpy())[:, None]
    floors = [fit_floor(cur[:, j]) for j in range(4)]
    t = np.column_stack([to_target(cur[:, j], floors[j]) for j in range(4)])
    sw = region_weights(case, splits.train, region_weight) if region_weight != 1.0 else None
    fam = make_family(family, seed=seed, **kw).fit(build_features(splits.train, kind), t, sample_weight=sw)
    info = {"family": family, "rows": len(splits.train), "kw": kw, "per_width": per_width, "region_weight": region_weight}
    return CaseModel(case, kind, [fam], floors, info, per_width=per_width)


def evaluate(model, df: pd.DataFrame) -> dict:
    """Per terminal: R² (in signed-log space, since currents span many decades), the same R²
    on linear currents, and the median / p95 / mean relative error."""
    pred, true = model.predict(df), df[list(spec.CURRENTS)].to_numpy()
    out = {}
    for j, term in enumerate(spec.CURRENTS):
        floor = fit_floor(true[:, j])
        err = summarise(rel_error(pred[:, j], true[:, j]))
        out[term] = {"r2": r2(to_target(pred[:, j], floor), to_target(true[:, j], floor)),
                     "r2_linear": r2(pred[:, j], true[:, j]), **err}
    return out


def select_best(candidates: Mapping, val: pd.DataFrame) -> str:
    """Name of the candidate with the lowest mean (over terminals) validation median error."""
    score = {name: np.mean([v["median"] for v in evaluate(m, val).values()]) for name, m in candidates.items()}
    return min(score, key=score.get)
