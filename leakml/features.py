"""Model inputs and the signed-log target transform."""
import numpy as np

from .cards import parse_pm
from .spec import L, PROCESS_PARAMS

_NOM = {}


def _nominal(kind):
    if not _NOM:
        for k, c in parse_pm().items():
            _NOM[k] = c.params
    return _NOM[kind]


FEATURE_NAMES = ("log_w", "inv_kT", "temp_c", "vdd", "vd", "vs", "vds", "vgs", "vbs") + \
    tuple(f"{p}_rel" for p in PROCESS_PARAMS)


def build_features(df, kind: str) -> np.ndarray:
    """Physics-aware inputs: log width, 1/kT, terminal voltages and differences,
    and each process parameter relative to its nominal value."""
    nom = _nominal(kind)
    cols = [np.log(df["w"].to_numpy() / L),
            11604.518 / (df["temp"].to_numpy() + 273.15),
            (df["temp"].to_numpy() - 27.0) / 100.0,
            df["vdd"].to_numpy(), df["vd"].to_numpy(), df["vs"].to_numpy(),
            (df["vd"] - df["vs"]).to_numpy(), (df["vg"] - df["vs"]).to_numpy(), (df["vb"] - df["vs"]).to_numpy()]
    cols += [df[p].to_numpy() / nom[p] - 1.0 for p in PROCESS_PARAMS]
    return np.column_stack(cols)


def fit_floor(currents) -> float:
    """Scale below which a current is treated as zero by the signed-log transform."""
    return max(1e-3 * float(np.median(np.abs(currents))), 1e-18)


def to_target(i, floor: float):
    i = np.asarray(i, float)
    return np.sign(i) * np.log1p(np.abs(i) / floor) / np.log(10.0)


def from_target(t, floor: float):
    t = np.asarray(t, float)
    return np.sign(t) * floor * np.expm1(np.abs(t) * np.log(10.0))


def weff(w):
    """Effective channel width in units of Lmin: BSIM4 uses W - 2*wint (wint = 5 nm in both cards).
    Currents are almost exactly proportional to it, so models predict current per weff."""
    wint = _nominal("nmos")["wint"]
    return (np.asarray(w, float) - 2.0 * wint) / L
