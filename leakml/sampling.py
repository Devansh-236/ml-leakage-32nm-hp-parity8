"""Seeded samplers for the PDF's section-4 distributions."""
import numpy as np
import pandas as pd

from . import spec
from .cards import parse_pm


def _nominal():
    return parse_pm()


def _process(rng, nom, n):
    """Process columns: nominal * (1 +/- 10 %), uniform, drawn in a fixed order."""
    out = {}
    for p in spec.PROCESS_PARAMS:
        lo, hi = sorted((nom[p] * (1 - spec.PROCESS_SPREAD), nom[p] * (1 + spec.PROCESS_SPREAD)))
        out[p] = rng.uniform(lo, hi, n)
    return out


EDGE_LAYER = 0.02       # e-folding thickness of the boundary layer, as a fraction of VDD


def _edge_ratio(rng, n: int) -> np.ndarray:
    """Terminal voltage / VDD concentrated at the rails: half exactly on a rail (0 or 1), the
    rest inside a thin layer (exponential, scale EDGE_LAYER) next to a rail."""
    rail = rng.integers(0, 2, n)
    offset = np.where(rng.random(n) < 0.5, 0.0, np.minimum(rng.exponential(EDGE_LAYER, n), 0.5))
    return np.where(rail == 1, 1.0 - offset, offset)


def draw_device_inputs(case: str, n: int, seed: int, edge_fraction: float = 0.0) -> pd.DataFrame:
    """The PDF's independent uniform draws. With edge_fraction > 0 that share of the Vd and Vs draws
    (chosen separately for each terminal) is replaced by boundary-layer draws (_edge_ratio). The
    default 0 is exactly the PDF's distribution and is used for the deliverable datasets."""
    kind, vg_mult, vb_mult = spec.CASES[case]
    rng = np.random.default_rng(seed)
    vdd = rng.uniform(*spec.VDD_RANGE, n)            # first: Vd, Vs depend on it
    ud, us = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    w = rng.uniform(*spec.W_RANGE, n)
    temp = rng.uniform(*spec.TEMP_RANGE, n)
    proc = _process(rng, _nominal()[kind].params, n)
    if edge_fraction > 0:                           # separate stream: the default draws stay untouched
        er = np.random.default_rng([seed, 1])
        ud = np.where(er.random(n) < edge_fraction, _edge_ratio(er, n), ud)
        us = np.where(er.random(n) < edge_fraction, _edge_ratio(er, n), us)
    vd, vs = ud * vdd, us * vdd
    return pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                         "vg": vdd * vg_mult, "vb": vdd * vb_mult, **proc})[list(spec.INPUT_COLS)]


def draw_corners(n: int, seed: int) -> pd.DataFrame:
    """Corners: one VDD, temperature, Wn, Wp and one process set per device kind."""
    rng = np.random.default_rng(seed)
    cards = _nominal()
    cols = {"corner_id": np.arange(n),
            "vdd": rng.uniform(*spec.VDD_RANGE, n),
            "temp": rng.uniform(*spec.TEMP_RANGE, n),
            "wn": rng.uniform(*spec.W_RANGE, n),
            "wp": rng.uniform(*spec.W_RANGE, n)}
    for prefix, kind in (("n", "nmos"), ("p", "pmos")):
        for p, v in _process(rng, cards[kind].params, n).items():
            cols[f"{prefix}_{p}"] = v
    return pd.DataFrame(cols)


def nominal_corner(vdd: float = spec.VDD_NOM, temp: float = 27.0,
                   wn: float = 2 * spec.L, wp: float = 2 * spec.L) -> dict:
    """A corner with every process parameter at its nominal value."""
    cards = _nominal()
    c = {"corner_id": 0, "vdd": vdd, "temp": temp, "wn": wn, "wp": wp}
    for prefix, kind in (("n", "nmos"), ("p", "pmos")):
        for p in spec.PROCESS_PARAMS:
            c[f"{prefix}_{p}"] = cards[kind].params[p]
    return c
