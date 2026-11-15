"""Single-transistor netlists, runs and (later) dataset generation."""
from typing import Mapping, Optional

from . import spice
from .cards import parse_pm
from . import spec
from .spec import CURRENTS, L, OPTIONS, PROCESS_PARAMS

_CARDS = {}


def cards():
    if "c" not in _CARDS:
        _CARDS["c"] = parse_pm()
    return _CARDS["c"]


def num(x) -> str:
    """Plain decimal text for SPICE (numpy scalars would print as np.float64(...))."""
    return repr(float(x))


def device_netlist(kind: str, p: Mapping[str, float]) -> str:
    card = cards()[kind].override({k: p[k] for k in PROCESS_PARAMS}, name="dut")
    return "\n".join([
        f"* single {kind.upper()} leakage sample",
        OPTIONS,
        f".options temp={num(p['temp'])}",
        card.render(),
        f"M1 d g s b dut W={num(p['w'])} L={num(L)}",
        f"Vd d 0 {num(p['vd'])}",
        f"Vg g 0 {num(p['vg'])}",
        f"Vs s 0 {num(p['vs'])}",
        f"Vb b 0 {num(p['vb'])}",
        ".control",
        "set numdgt=12",
        "op",
        "print i(vd) i(vg) i(vs) i(vb)",
        ".endc",
        ".end",
        "",
    ])


def parse_device(stdout: str) -> Optional[dict]:
    vals = spice.parse_prints(stdout)
    keys = [f"i(v{t[1]})" for t in CURRENTS]
    if not all(k in vals for k in keys):
        return None
    return {t: vals[k] for t, k in zip(CURRENTS, keys)}


def run_device(kind: str, p: Mapping[str, float]) -> Optional[dict]:
    """Currents through the four terminal sources {id, ig, is, ib} (positive out of the
    device), or None if the simulation failed."""
    return parse_device(spice.run_netlist(device_netlist(kind, p)).stdout)


def _simulate(kind: str, df, workers=None) -> list:
    """Simulate every row of an input frame; one dict (or None on failure) per row."""
    texts = [device_netlist(kind, row) for row in df.to_dict("records")]
    return [parse_device(r.stdout) for r in spice.run_many(texts, workers)]


def generate_case(case: str, n: int, seed: int, workers=None, edge_fraction: float = 0.0):
    """n clean simulated rows for a case (columns spec.DATA_COLS) and the number of
    failed simulations that had to be redrawn. Deterministic for a given seed. `edge_fraction` > 0
    draws part of Vd/Vs next to the rails (training-only data; see sampling.draw_device_inputs)."""
    import pandas as pd

    from .sampling import draw_device_inputs

    kind = spec.CASES[case][0]
    rows, failed, rnd = [], 0, 0
    need = n
    while need > 0:
        inputs = draw_device_inputs(case, need, seed + 1_000_003 * rnd, edge_fraction)
        res = _simulate(kind, inputs, workers)
        for rec, cur in zip(inputs.to_dict("records"), res):
            if cur is None:
                failed += 1
            else:
                rows.append({**rec, **cur})
        need = n - len(rows)
        rnd += 1
        if rnd > 50:
            raise RuntimeError(f"{case}: more than 50 redraw rounds, simulations keep failing")
    return pd.DataFrame(rows)[list(spec.DATA_COLS)], failed
