"""Circuit leakage from the four transistor models (PDF Part B step 4, Part C step 7).

Terminal currents are positive out of the device. For each transistor the four models
give (id, ig, is, ib) from its terminal voltages. The leakage is the current delivered by
the VDD source (plus the current delivered by inputs held at VDD), obtained from
Kirchhoff's current law over a node set S that contains VDD:

    delivered by VDD = - sum of the device-terminal currents located at nodes of S

A device whose drain and source are both in S contributes id + is = -(ig + ib) by its
own KCL, so its channel current cancels and only its gate / body terminals outside S
remain. That matters because an ON device with Vds of microvolts has a channel current
that is a tiny difference, which no regression model resolves in relative terms.

S starts as VDD plus every node reachable from it through ON channels (so no ON device is
cut), and may be extended by further internal nodes. S is valid if every device with
exactly one of drain/source in S is OFF. Among valid S the one whose cut devices have the
largest smallest |Vds| is used: there a cut device's channel current dominates its gate
tunnelling current.
"""
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .circuits import Circuit
from .spec import CURRENTS, INPUT_COLS, PROCESS_PARAMS

RAILS = ("vdd", "0")


def _is_on(kind: str, vg: np.ndarray, vdd: np.ndarray) -> np.ndarray:
    return vg > 0.5 * vdd if kind == "nmos" else vg < 0.5 * vdd


def device_inputs(f, nodes: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """Model-input columns (spec.INPUT_COLS) of one transistor across all rows."""
    pre = f.kind[0]
    df = pd.DataFrame({"vdd": corners["vdd"].to_numpy(), "temp": corners["temp"].to_numpy(),
                       "w": corners["wn" if f.kind == "nmos" else "wp"].to_numpy(),
                       "vd": nodes[f.d].to_numpy(), "vs": nodes[f.s].to_numpy(),
                       "vg": nodes[f.g].to_numpy(), "vb": nodes[f.b].to_numpy()})
    for p in PROCESS_PARAMS:
        df[p] = corners[f"{pre}_{p}"].to_numpy()
    return df[list(INPUT_COLS)]


def device_table(c: Circuit, nodes: pd.DataFrame, corners: pd.DataFrame, models: Mapping):
    """({fet name: (n, 4) currents id, ig, is, ib}, (n, n_fets) boolean ON matrix)."""
    cur, on = {}, []
    for f in c.fets:
        df = device_inputs(f, nodes, corners)
        is_on = _is_on(f.kind, df["vg"].to_numpy(), df["vdd"].to_numpy())
        out = np.zeros((len(df), 4))
        for flag in (True, False):
            sel = is_on == flag
            if sel.any():
                out[sel] = models[f"{f.kind.upper()}_{'ON' if flag else 'OFF'}"].predict(df[sel].reset_index(drop=True))
        cur[f.name] = out
        on.append(is_on)
    return cur, np.column_stack(on)


def _reachable(c: Circuit, on: Sequence[bool]) -> set:
    s, grew = {"vdd"}, True
    while grew:
        grew = False
        for f, o in zip(c.fets, on):
            if o and ((f.d in s) != (f.s in s)):
                s |= {f.d, f.s}
                grew = True
    return s


def choose_cut(c: Circuit, on: Sequence[bool], vref: Mapping[str, float]) -> frozenset:
    """Node set S (see module docstring) for one ON/OFF pattern; `vref` maps node -> typical voltage."""
    base = _reachable(c, on)
    channel = {n for f in c.fets for n in (f.d, f.s)}
    gate_only = {f.g for f in c.fets} - channel
    free = sorted(channel - base - set(RAILS) - gate_only)
    best, best_score = frozenset(base), -1.0
    for k in range(len(free) + 1):
        for extra in combinations(free, k):
            s = base | set(extra)
            cut = [(f, o) for f, o in zip(c.fets, on) if (f.d in s) != (f.s in s)]
            if not cut or any(o for _, o in cut):
                continue
            score = min(abs(vref[f.d] - vref[f.s]) for f, _ in cut)
            if score > best_score + 1e-12:
                best, best_score = frozenset(s), score
    return best


def _supply_parts(c: Circuit, S: frozenset, cur: Mapping) -> dict:
    """The supply current of `_supply_current`, split by which terminal carries it:
    channel (drain / source), gate and body."""
    ch = ga = bo = 0.0
    for f in c.fets:
        i = cur[f.name]                       # columns: id, ig, is, ib
        if f.d in S and f.s in S:
            ga = ga + i[:, 1] * (f.g not in S)
            bo = bo + i[:, 3] * (f.b not in S)
        else:
            ch = ch - (i[:, 0] * (f.d in S) + i[:, 2] * (f.s in S))
            ga = ga - i[:, 1] * (f.g in S)
            bo = bo - i[:, 3] * (f.b in S)
    return {"channel": ch, "gate": ga, "body": bo}


def _supply_current(c: Circuit, S: frozenset, cur: Mapping) -> np.ndarray:
    p = _supply_parts(c, S, cur)
    return p["channel"] + p["gate"] + p["body"]


def _pin_current(c: Circuit, node: str, cur: Mapping) -> np.ndarray:
    """Current delivered by the source holding `node` (input pin)."""
    total = 0.0
    for f in c.fets:
        i = cur[f.name]
        for k, t in enumerate((f.d, f.g, f.s, f.b)):
            if t == node:
                total = total - i[:, k]
    return total


def ml_leakage(c: Circuit, nodes: pd.DataFrame, corners: pd.DataFrame, models: Mapping) -> np.ndarray:
    """Total leakage (VDD supply current + current delivered by inputs held at VDD), per row."""
    cur, on = device_table(c, nodes, corners, models)
    vdd = corners["vdd"].to_numpy()
    leak = np.zeros(len(nodes))
    for pat in {tuple(r) for r in on}:
        sel = (on == np.array(pat)).all(axis=1)
        S = choose_cut(c, pat, nodes[sel].median().to_dict())
        sub = {k: v[sel] for k, v in cur.items()}
        leak[sel] = _supply_current(c, S, sub)
    for n in c.inputs:
        high = nodes[n].to_numpy() > 0.5 * vdd
        leak += np.where(high, _pin_current(c, n, cur), 0.0)
    return leak


def leakage_parts(c: Circuit, nodes: pd.DataFrame, corners: pd.DataFrame, models: Mapping) -> dict:
    """ml_leakage split into channel / gate / body parts (input-pin currents count as gate current)."""
    cur, on = device_table(c, nodes, corners, models)
    vdd = corners["vdd"].to_numpy()
    parts = {k: np.zeros(len(nodes)) for k in ("channel", "gate", "body")}
    for pat in {tuple(r) for r in on}:
        sel = (on == np.array(pat)).all(axis=1)
        S = choose_cut(c, pat, nodes[sel].median().to_dict())
        for k, v in _supply_parts(c, S, {n: x[sel] for n, x in cur.items()}).items():
            parts[k][sel] += v
    for n in c.inputs:
        parts["gate"] += np.where(nodes[n].to_numpy() > 0.5 * vdd, _pin_current(c, n, cur), 0.0)
    return parts


def solve_internal(c: Circuit, nodes: pd.DataFrame, corners: pd.DataFrame, models: Mapping,
                   unknown: Sequence[str], iters: int = 30, sweeps: int = 2) -> pd.DataFrame:
    """Copy of `nodes` where each node in `unknown` is set to the voltage in [0, VDD] at
    which the model currents into it sum to zero (vectorised bisection; the net current
    delivered by the devices decreases as the node voltage rises). Several sweeps when more
    than one node is unknown, since they are coupled."""
    nodes = nodes.copy()
    vdd = corners["vdd"].to_numpy()

    def residual(node):
        fets = [f for f in c.fets if node in (f.d, f.g, f.s, f.b)]
        sub = Circuit(c.name, c.inputs, tuple(fets), c.internal, c.output)
        cur, _ = device_table(sub, nodes, corners, models)
        return -_pin_current(sub, node, cur)       # sum of terminal currents out of devices into the node

    for _ in range(sweeps if len(unknown) > 1 else 1):
        for node in unknown:
            lo, hi = np.zeros(len(nodes)), vdd.copy()
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                nodes[node] = mid
                above = residual(node) > 0         # net current still flowing in: voltage should be higher
                lo = np.where(above, mid, lo)
                hi = np.where(above, hi, mid)
            nodes[node] = 0.5 * (lo + hi)
    return nodes
