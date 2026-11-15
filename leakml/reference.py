"""SPICE reference runs for Part B (gates) and Part C (the parity checker)."""
import itertools
from typing import Iterable, Optional

import pandas as pd

from . import spice
from .circuits import GATES, circuit_netlist, flatten, net_values, parse_circuit, simulate_circuit
from .parity import PARITY8, vector_bits


def _run(circuits_bits_corners, workers):
    """Simulate [(Circuit, corner dict, bits)]; raise if any run fails."""
    texts = [circuit_netlist(c, corner, bits) for c, corner, bits in circuits_bits_corners]
    out, failed = [], 0
    for (c, corner, bits), r in zip(circuits_bits_corners, spice.run_many(texts, workers)):
        res = parse_circuit(c, r.stdout, bits)
        failed += res is None
        out.append(res)
    if failed:
        raise RuntimeError(f"{failed} of {len(out)} SPICE runs failed")
    return out


def gate_reference(corners: pd.DataFrame, workers: Optional[int] = None) -> pd.DataFrame:
    """Every gate and input pattern at every corner (18 cases per corner)."""
    jobs, meta = [], []
    for corner in corners.to_dict("records"):
        for gate, c in GATES.items():
            for bits in itertools.product((0, 1), repeat=len(c.inputs)):
                jobs.append((c, corner, bits))
                meta.append((corner["corner_id"], gate, "".join(map(str, bits))))
    rows = []
    for (cid, gate, pat), r in zip(meta, _run(jobs, workers)):
        row = {"corner_id": cid, "gate": gate, "pattern": pat, "leak": r["leak"], "iddd": r["iddd"],
               "x": r.get("x"), "y": r.get("y"), "n": r.get("n"), "out": r["out"]}
        row.update({k: v for k, v in r.items() if k.startswith("pin_")})
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({"pattern": str, "x": float, "y": float, "n": float})


def parity_reference(corners: pd.DataFrame, vectors: Iterable[int] = range(256),
                     workers: Optional[int] = None, circuit=None) -> pd.DataFrame:
    """The whole circuit (default PARITY8) simulated as one netlist, every vector at every corner."""
    flat = flatten(circuit or PARITY8)
    vectors = list(vectors)
    jobs, meta = [], []
    for corner in corners.to_dict("records"):
        for v in vectors:
            jobs.append((flat, corner, vector_bits(v)))
            meta.append((corner["corner_id"], v))
    rows = [{"corner_id": cid, "vector": v, "leak": r["leak"], "iddd": r["iddd"], "p": r["p"]}
            for (cid, v), r in zip(meta, _run(jobs, workers))]
    return pd.DataFrame(rows)


def gate_sum_leakage(comp, corner, bits) -> float:
    """Sum of the separately simulated leakages of every gate instance of a composite, each
    with its inputs at the logic values of this vector (the PDF's composition assumption)."""
    v = net_values(comp, bits)
    return sum(simulate_circuit(GATES[i.gate], corner, tuple(v[n] for n in i.inputs))["leak"]
               for i in comp.instances)
