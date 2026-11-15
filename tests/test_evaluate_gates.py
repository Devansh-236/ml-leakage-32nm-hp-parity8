import numpy as np
import pytest

from leakml.circuits import logic
from leakml.evaluate import gate_predictions, gate_table, pooled
from leakml.oracle import OracleModels
from leakml.reference import gate_reference
from leakml.sampling import draw_corners


@pytest.fixture(scope="module")
def setup():
    corners = draw_corners(4, seed=31)
    return corners, gate_reference(corners, workers=4), OracleModels()


def test_variant_a_with_oracle(setup):
    corners, spice, models = setup
    p = gate_predictions(models, corners, spice, "a")
    assert len(p) == 18 * 4 and set(p.variant) == {"a"}
    assert pooled(p)["mean"] < 1e-3


def test_variant_b_with_oracle_and_rails(setup):
    corners, spice, models = setup
    p = gate_predictions(models, corners, spice, "b", iters=18)
    assert pooled(p)["mean"] < 5e-3
    vdd = corners.set_index("corner_id").vdd.loc[p.corner_id].to_numpy()
    expect = np.array([logic(g, [int(ch) for ch in pat]) for g, pat in zip(p.gate, p.pattern)]) * vdd
    assert p.ml_out.to_numpy() == pytest.approx(expect, rel=1e-12)
    assert p[p.gate == "NAND2"].ml_x.notna().all() and p[p.gate == "INV"].ml_x.isna().all()


def test_table_has_18_rows_and_columns(setup):
    corners, spice, models = setup
    t = gate_table(gate_predictions(models, corners, spice, "a"))
    assert len(t) == 18
    assert {"gate", "pattern", "spice_median", "ml_median", "bias", "median_err", "p95_err", "mean_err"} <= set(t.columns)
