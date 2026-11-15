"""ML-versus-SPICE evaluation of gates (Part B) and the parity checker (Part C)."""
from typing import Mapping

import numpy as np
import pandas as pd

from .circuits import GATES, logic
from .compose import ml_leakage, solve_internal
from .metrics import bias, rel_error, summarise

# the node solved by the transistor models for each gate (the series-stack node)
STACK_NODE = {"NAND2": "x", "NOR2": "y", "AND2": "x", "OR2": "y"}


def _bits(pattern: str) -> tuple:
    return tuple(int(ch) for ch in pattern)


def ml_gate_nodes(gate: str, patterns, corners: pd.DataFrame, models: Mapping, iters: int = 30) -> pd.DataFrame:
    """Node voltages of a gate without any SPICE node data: inputs at 0/VDD, the gate output
    (and the first-stage output of AND2/OR2) at its logic rail, the series-stack node solved
    by the transistor models. `corners` has one row per entry of `patterns`."""
    c = GATES[gate]
    vdd = corners["vdd"].to_numpy()
    bits = [_bits(p) for p in patterns]
    nodes = pd.DataFrame({"vdd": vdd, "0": 0.0})
    for k, n in enumerate(c.inputs):
        nodes[n] = vdd * np.array([b[k] for b in bits])
    nodes["out"] = vdd * np.array([logic(gate, b) for b in bits])
    if "n" in c.internal:                                   # AND2 / OR2: output of the NAND2 / NOR2 stage
        first = "NAND2" if gate == "AND2" else "NOR2"
        nodes["n"] = vdd * np.array([logic(first, b) for b in bits])
    if gate in STACK_NODE:
        node = STACK_NODE[gate]
        nodes[node] = 0.5 * vdd
        nodes = solve_internal(c, nodes, corners, models, [node], iters=iters)
    return nodes


def gate_predictions(models: Mapping, corners: pd.DataFrame, spice: pd.DataFrame, variant: str,
                     iters: int = 30) -> pd.DataFrame:
    """ML leakage of every gate case in `spice` (columns corner_id, gate, pattern, leak, x, y, n, out).
    variant "a": internal and output node voltages recorded from SPICE (PDF step 4);
    variant "b": internal nodes solved by the transistor models, output at the logic rail."""
    assert variant in ("a", "b")
    corner_by_id = corners.set_index("corner_id", drop=False)
    out = []
    for gate, c in GATES.items():
        s = spice[spice.gate == gate].reset_index(drop=True)
        if s.empty:
            continue
        crn = corner_by_id.loc[s.corner_id].reset_index(drop=True)
        vdd = crn["vdd"].to_numpy()
        if variant == "a":
            nodes = pd.DataFrame({"vdd": vdd, "0": 0.0})
            for k, n in enumerate(c.inputs):
                nodes[n] = vdd * np.array([_bits(p)[k] for p in s.pattern])
            for n in c.internal + (c.output,):
                nodes[n] = s[n].to_numpy()
        else:
            nodes = ml_gate_nodes(gate, list(s.pattern), crn, models, iters=iters)
        ml = ml_leakage(c, nodes, crn, models)
        res = pd.DataFrame({"corner_id": s.corner_id, "gate": gate, "pattern": s.pattern.astype(str), "variant": variant,
                            "spice_leak": s.leak, "ml_leak": ml, "err": rel_error(ml, s.leak.to_numpy()),
                            "spice_out": s.out, "ml_out": nodes["out"].to_numpy()})
        for n in ("x", "y", "n"):
            res[f"spice_{n}"] = s[n] if n in c.internal else np.nan
            res[f"ml_{n}"] = nodes[n].to_numpy() if n in c.internal else np.nan
        out.append(res)
    return pd.concat(out, ignore_index=True)


def gate_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, gate, input pattern): SPICE median, ML median, bias, median / p95 / mean error."""
    rows = []
    for (variant, gate, pat), g in pred.groupby(["variant", "gate", "pattern"], sort=False):
        s = summarise(g.err)
        rows.append({"variant": variant, "gate": gate, "pattern": pat, "spice_median": g.spice_leak.median(),
                     "ml_median": g.ml_leak.median(), "bias": bias(g.ml_leak, g.spice_leak),
                     "median_err": s["median"], "p95_err": s["p95"], "mean_err": s["mean"],
                     "within_1pct": s["within_1pct"], "max_err": s["max"]})
    t = pd.DataFrame(rows)
    return t if pred.variant.nunique() > 1 else t.drop(columns="variant").reset_index(drop=True)


def pooled(pred: pd.DataFrame) -> dict:
    return summarise(pred.err)


def ml_gate_table(models: Mapping, corners: pd.DataFrame, iters: int = 30) -> pd.DataFrame:
    """ML leakage of all 18 gate cases at every corner, with no SPICE node data
    (columns corner_id, gate, pattern, ml_leak)."""
    import itertools
    out = []
    for gate, c in GATES.items():
        pats = ["".join(map(str, b)) for b in itertools.product((0, 1), repeat=len(c.inputs))]
        crn = pd.concat([corners] * len(pats), ignore_index=True)
        patterns = [p for p in pats for _ in range(len(corners))]
        nodes = ml_gate_nodes(gate, patterns, crn, models, iters=iters)
        out.append(pd.DataFrame({"corner_id": crn["corner_id"], "gate": gate, "pattern": patterns,
                                 "ml_leak": ml_leakage(c, nodes, crn, models)}))
    return pd.concat(out, ignore_index=True)


def parity_predictions(models: Mapping, corners: pd.DataFrame, spice=None, iters: int = 30,
                       vectors=range(256), circuit=None) -> pd.DataFrame:
    """ML leakage of the parity checker: for each corner and input vector, the sum of the gate
    models' leakages with every gate's inputs at its logic values (loading between gates ignored).
    Gate leakage depends only on (corner, gate, input pattern), so the 18 cases are evaluated
    once per corner and looked up per instance. `spice` (columns corner_id, vector, leak) adds
    spice_leak and err. `circuit` is the gate-level composite (default PARITY8)."""
    from .circuits import net_values
    from .parity import PARITY8, vector_bits
    circuit = circuit or PARITY8

    table = ml_gate_table(models, corners, iters)
    case_ids = {k: i for i, k in enumerate(dict.fromkeys(zip(table.gate, table.pattern)))}
    ids = corners["corner_id"].to_numpy()
    mat = np.zeros((len(ids), len(case_ids)))
    row_of = {cid: r for r, cid in enumerate(ids)}
    for cid, gate, pat, leak in zip(table.corner_id, table.gate, table.pattern, table.ml_leak):
        mat[row_of[cid], case_ids[(gate, pat)]] = leak
    vectors = list(vectors)
    pick = []
    for v in vectors:
        nets = net_values(circuit, vector_bits(v))
        pick.append([case_ids[(i.gate, "".join(str(nets[n]) for n in i.inputs))] for i in circuit.instances])
    pick = np.array(pick)                                  # (vectors, instances)
    leak = mat[:, pick].sum(axis=2)                        # (corners, vectors)
    res = pd.DataFrame({"corner_id": np.repeat(ids, len(vectors)), "vector": np.tile(vectors, len(ids)),
                        "ml_leak": leak.ravel()})
    if spice is not None:
        res = res.merge(spice[["corner_id", "vector", "leak"]].rename(columns={"leak": "spice_leak"}),
                        on=["corner_id", "vector"], how="left")
        res["err"] = rel_error(res.ml_leak, res.spice_leak)
    return res


def parity_table(pred: pd.DataFrame) -> pd.DataFrame:
    """By number of ones in the input vector (0..8) and overall: SPICE / ML median, bias and errors."""
    pred = pred.assign(ones=pred.vector.map(lambda v: bin(int(v)).count("1")))
    rows = {}
    groups = [(str(k), g) for k, g in pred.groupby("ones")] + [("all", pred)]
    for name, g in groups:
        s = summarise(g.err)
        rows[name] = {"spice_median": g.spice_leak.median(), "ml_median": g.ml_leak.median(),
                      "bias": bias(g.ml_leak, g.spice_leak), "median_err": s["median"], "p95_err": s["p95"],
                      "mean_err": s["mean"], "within_1pct": s["within_1pct"], "max_err": s["max"], "n": len(g)}
    return pd.DataFrame(rows).T
