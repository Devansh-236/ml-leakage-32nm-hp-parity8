import numpy as np
import pytest

from leakml.evaluate import parity_predictions, parity_table
from leakml.oracle import OracleModels
from leakml.reference import parity_reference
from leakml.sampling import draw_corners


@pytest.fixture(scope="module")
def models():
    return OracleModels()


def test_parity_with_oracle_matches_circuit_spice(models):
    corners = draw_corners(2, seed=41)
    vectors = range(0, 256, 32)
    spice = parity_reference(corners, vectors=vectors, workers=4)
    p = parity_predictions(models, corners, spice, iters=18, vectors=vectors)
    assert len(p) == 16
    assert p.err.max() < 0.03          # gate-sum composition error (SPICE circuit vs sum of gates) plus node solve


def test_no_circuit_spice_needed_and_vector_count(models):
    corners = draw_corners(1, seed=42)
    p = parity_predictions(models, corners, None, iters=14)
    assert len(p) == 256 and set(p.columns) >= {"corner_id", "vector", "ml_leak"}
    assert np.isfinite(p.ml_leak).all() and (p.ml_leak > 0).all()
    assert "err" not in p.columns


def test_parity_table_groups(models):
    corners = draw_corners(1, seed=43)
    spice = parity_reference(corners, vectors=range(0, 256, 16), workers=4)
    t = parity_table(parity_predictions(models, corners, spice, iters=14, vectors=range(0, 256, 16)))
    assert t.index.tolist()[-1] == "all" and len(t) >= 3
    assert {"spice_median", "ml_median", "bias", "median_err", "p95_err", "mean_err"} <= set(t.columns)


def test_nand_topology_with_oracle_matches_circuit_spice(models):
    from leakml.parity import PARITY8_NAND
    corners = draw_corners(2, seed=44)
    vectors = range(3, 256, 37)
    spice = parity_reference(corners, vectors=vectors, workers=4, circuit=PARITY8_NAND)
    assert ((spice.p > 0.5).astype(int).to_numpy() ==
            np.array([bin(v).count("1") % 2 for v in spice.vector])).all()
    p = parity_predictions(models, corners, spice, iters=18, vectors=vectors, circuit=PARITY8_NAND)
    assert len(p) == 2 * len(vectors) and p.err.max() < 0.03


def test_tables_report_tail_metrics():
    import pandas as pd
    from leakml.evaluate import gate_table
    pred = pd.DataFrame({"variant": ["a"] * 4, "gate": ["INV"] * 4, "pattern": ["0"] * 4,
                         "spice_leak": [1.0] * 4, "ml_leak": [1.0, 1.005, 1.02, 1.5], "err": [0.0, 0.005, 0.02, 0.5]})
    g = gate_table(pred).iloc[0]
    assert g.within_1pct == pytest.approx(0.5) and g.max_err == pytest.approx(0.5)
    par = pd.DataFrame({"vector": [1, 3, 3, 7], "spice_leak": [1.0] * 4, "ml_leak": [1.0] * 4, "err": [0.0, 0.02, 0.005, 0.3]})
    t = parity_table(par)
    assert t.loc["all", "within_1pct"] == pytest.approx(0.5) and t.loc["all", "max_err"] == pytest.approx(0.3)
