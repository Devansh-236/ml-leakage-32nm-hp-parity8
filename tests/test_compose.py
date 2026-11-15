import itertools

import numpy as np
import pandas as pd
import pytest

from leakml import spec
from leakml.circuits import GATES, logic, simulate_circuit
from leakml.compose import choose_cut, device_table, ml_leakage, solve_internal
from leakml.oracle import OracleModels
from leakml.sampling import draw_corners


@pytest.fixture(scope="module")
def models():
    return OracleModels()


def gate_frames(gate, patterns, corners):
    """nodes / corners frames (one row per pattern x corner) from SPICE runs, plus SPICE leakage."""
    c = GATES[gate]
    rows, crows, leaks = [], [], []
    for corner in corners.to_dict("records"):
        for bits in patterns:
            r = simulate_circuit(c, corner, bits)
            row = {"vdd": corner["vdd"], "0": 0.0}
            for n, b in zip(c.inputs, bits):
                row[n] = corner["vdd"] * b
            for n in c.internal + (c.output,):
                row[n] = r[n]
            rows.append(row)
            crows.append(corner)
            leaks.append(r["leak"])
    return c, pd.DataFrame(rows), pd.DataFrame(crows).reset_index(drop=True), np.array(leaks)


ALL = [(g, bits) for g, c in GATES.items() for bits in itertools.product((0, 1), repeat=len(c.inputs))]


def test_oracle_kcl_matches_spice_gates(models):
    corners = draw_corners(5, seed=21)
    worst = 0.0
    for gate in GATES:
        pats = list(itertools.product((0, 1), repeat=len(GATES[gate].inputs)))
        c, nodes, crn, spice_leak = gate_frames(gate, pats, corners)
        ml = ml_leakage(c, nodes, crn, models)
        worst = max(worst, float(np.max(np.abs(ml - spice_leak) / spice_leak)))
    assert worst < 1e-3, worst


def test_cut_avoids_on_devices(models):
    corners = draw_corners(3, seed=22)
    c, nodes, crn, _ = gate_frames("NOR2", [(1, 1)], corners)
    cur, on = device_table(c, nodes, crn, models)
    S = choose_cut(c, on[0], nodes.median().to_dict())
    assert "vdd" in S and "y" in S and "out" not in S         # cuts through the lower PMOS, not the upper
    cut = [(f, o) for f, o in zip(c.fets, on[0]) if (f.d in S) != (f.s in S)]
    assert cut and not any(o for _, o in cut)


def test_solver_recovers_nodes(models):
    corners = draw_corners(2, seed=23)
    for gate in ("NAND2", "NOR2", "AND2", "OR2"):
        pats = list(itertools.product((0, 1), repeat=2))
        c, nodes, crn, _ = gate_frames(gate, pats, corners)
        unknown = [n for n in c.internal if n in ("x", "y")]
        start = nodes.copy()
        for u in unknown:
            start[u] = 0.5 * start["vdd"]                    # wrong on purpose
        solved = solve_internal(c, start, crn, models, unknown, iters=14)
        for u in unknown:
            assert np.max(np.abs(solved[u] - nodes[u])) < 0.005, (gate, u)


def test_vds_zero_gives_finite_values(models):
    corners = draw_corners(2, seed=24)
    c, nodes, crn, _ = gate_frames("NAND2", [(1, 0)], corners)
    nodes["x"] = nodes["out"]                                # vd == vs exactly for the top NMOS
    assert np.isfinite(ml_leakage(c, nodes, crn, models)).all()
