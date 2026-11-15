import numpy as np
import pytest

from leakml import spec
from leakml.cards import parse_pm
from leakml.sampling import draw_corners, draw_device_inputs

CASES = list(spec.CASES)


@pytest.mark.parametrize("case", CASES)
def test_ranges(case):
    kind, vg_m, vb_m = spec.CASES[case]
    df = draw_device_inputs(case, 10000, seed=1)
    assert list(df.columns) == list(spec.INPUT_COLS)
    assert df.vdd.between(*spec.VDD_RANGE).all()
    assert df.temp.between(*spec.TEMP_RANGE).all()
    assert df.w.between(*spec.W_RANGE).all()
    assert (df.vg == df.vdd * vg_m).all() and (df.vb == df.vdd * vb_m).all()
    nom = parse_pm()[kind].params
    for p in spec.PROCESS_PARAMS:
        lo, hi = sorted((nom[p] * 0.9, nom[p] * 1.1))
        assert df[p].between(lo, hi).all(), p
        assert df[p].std() > 0.03 * abs(nom[p])


def test_seed_reproducible_and_distinct():
    a = draw_device_inputs("NMOS_OFF", 50, seed=7)
    b = draw_device_inputs("NMOS_OFF", 50, seed=7)
    c = draw_device_inputs("NMOS_OFF", 50, seed=8)
    assert a.equals(b) and not a.equals(c)


def test_vd_vs_follow_vdd():
    df = draw_device_inputs("PMOS_ON", 10000, seed=2)
    assert (df.vd <= df.vdd).all() and (df.vs <= df.vdd).all()
    assert (df.vd >= 0).all() and (df.vs >= 0).all()
    assert (df.vd > 0.9).any()


def test_corners_shapes():
    c = draw_corners(200, seed=3)
    assert len(c) == 200
    assert c.wn.between(*spec.W_RANGE).all() and c.wp.between(*spec.W_RANGE).all()
    assert c.vdd.between(*spec.VDD_RANGE).all()
    big = draw_corners(5000, seed=4)
    assert abs(np.corrcoef(big.n_vth0, big.p_vth0)[0, 1]) < 0.2
    assert list(big.columns[:5]) == ["corner_id", "vdd", "temp", "wn", "wp"]


def test_edge_fraction_zero_is_the_pdf_distribution():
    a = draw_device_inputs("NMOS_OFF", 500, seed=9)
    b = draw_device_inputs("NMOS_OFF", 500, seed=9, edge_fraction=0.0)
    assert a.equals(b)


@pytest.mark.parametrize("case", CASES)
def test_edge_draws_concentrate_at_the_rails(case):
    kind, vg_m, vb_m = spec.CASES[case]
    df = draw_device_inputs(case, 20000, seed=3, edge_fraction=1.0)
    assert list(df.columns) == list(spec.INPUT_COLS)
    assert (df.vd >= 0).all() and (df.vd <= df.vdd).all() and (df.vs >= 0).all() and (df.vs <= df.vdd).all()
    for col in ("vd", "vs"):
        rail = np.minimum(df[col], df.vdd - df[col]) / df.vdd
        assert (rail < 0.001).mean() > 0.4                 # about half sit exactly on a rail
        assert (rail < 0.10).mean() > 0.9                  # nearly all within 10 % of a rail
    exact = ((df.vd == 0) | (df.vd == df.vdd)) & ((df.vs == 0) | (df.vs == df.vdd))
    assert exact.mean() > 0.15                             # the four corners are well covered
    assert (df.vg == df.vdd * vg_m).all() and (df.vb == df.vdd * vb_m).all()
    assert df.vdd.between(*spec.VDD_RANGE).all() and df.temp.between(*spec.TEMP_RANGE).all()


def test_edge_fraction_mixes_with_uniform():
    df = draw_device_inputs("NMOS_ON", 20000, seed=4, edge_fraction=0.5)
    mid = ((df.vd / df.vdd).between(0.3, 0.7)).mean()
    assert 0.1 < mid < 0.3                                 # uniform half contributes about 0.5 * 0.4


@pytest.mark.parametrize("case", CASES)
def test_default_draw_reproduces_the_committed_dataset(case):
    """The edge-sampling refactor must not change the PDF datasets: the inputs of the committed
    CSV are exactly what the default sampler draws for the data seed."""
    import pandas as pd
    csv = pd.read_csv(spec.ROOT / "data" / "partA" / f"{case}.csv")
    drawn = draw_device_inputs(case, 10000, spec.SEEDS["data"])
    for col in spec.INPUT_COLS:
        assert csv[col].to_numpy() == pytest.approx(drawn[col].to_numpy(), rel=1e-12, abs=0), col
