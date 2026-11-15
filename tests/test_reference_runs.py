import numpy as np
import pytest

from leakml import spec
from leakml.circuits import GATES, flatten
from leakml.parity import PARITY8, vector_bits
from leakml.reference import gate_reference, gate_sum_leakage, parity_reference
from leakml.sampling import draw_corners


def test_vector_bits():
    assert vector_bits(0) == (0,) * 8
    assert vector_bits(1) == (1, 0, 0, 0, 0, 0, 0, 0)
    assert vector_bits(255) == (1,) * 8 and vector_bits(6) == (0, 1, 1, 0, 0, 0, 0, 0)


def test_gate_reference_small():
    df = gate_reference(draw_corners(3, 11), workers=2)
    assert len(df) == 18 * 3
    assert {"corner_id", "gate", "pattern", "leak", "iddd", "x", "y", "n", "out"} <= set(df.columns)
    assert (df.leak > 0).all() and not df[["leak", "iddd", "out"]].isna().any().any()
    assert df.groupby("gate").pattern.nunique().to_dict() == {"AND2": 4, "INV": 2, "NAND2": 4, "NOR2": 4, "OR2": 4}
    assert df[df.gate == "INV"].x.isna().all() and df[df.gate == "AND2"].x.notna().all()


def test_parity_reference_small():
    df = parity_reference(draw_corners(1, 12), vectors=range(0, 256, 40), workers=2)
    assert len(df) == 7 and (df.leak > 0).all()
    assert (abs(df.p - df.vector.map(lambda v: bin(v).count("1") % 2)) < 0.1).all()


def test_circuit_vs_gate_sum():
    flat = flatten(PARITY8)
    corners = draw_corners(3, 13)
    from leakml.circuits import simulate_circuit
    for corner in corners.to_dict("records"):
        for v in (0, 1, 85, 170, 255):
            bits = vector_bits(v)
            whole = simulate_circuit(flat, corner, bits)["leak"]
            parts = gate_sum_leakage(PARITY8, corner, bits)
            assert parts == pytest.approx(whole, rel=0.03), (corner["corner_id"], v)


def test_report_and_dev_corners_differ():
    a = draw_corners(spec.N_CORNERS, spec.SEEDS["corners_report"])
    b = draw_corners(spec.N_CORNERS, spec.SEEDS["corners_dev"])
    assert len(set(zip(a.vdd.round(9), a.temp.round(6))) & set(zip(b.vdd.round(9), b.temp.round(6)))) == 0
