import numpy as np
import pytest

from leakml.features import build_features, fit_floor, from_target, to_target
from leakml.metrics import bias, r2, rel_error, summarise
from leakml.sampling import draw_device_inputs


def test_rel_error_basic():
    assert rel_error(np.array([1.01]), np.array([1.0]))[0] == pytest.approx(0.01)


def test_rel_error_zero_true_is_finite():
    e = rel_error(np.array([1e-20, 0.0]), np.array([0.0, 0.0]))
    assert np.isfinite(e).all()


def test_summarise_keys_and_values():
    s = summarise(np.array([0.0, 0.01, 0.02, 0.03, 1.0]))
    assert set(s) >= {"median", "p95", "mean", "accuracy"}
    assert s["median"] == pytest.approx(0.02)
    assert s["accuracy"] == pytest.approx(1 - s["mean"])


def test_summarise_reports_tail_metrics():
    s = summarise(np.array([0.0, 0.01, 0.02, 0.03, 1.0]))
    assert s["within_1pct"] == pytest.approx(0.4)          # 0.0 and 0.01 are within 1 %
    assert s["max"] == pytest.approx(1.0)


def test_r2_and_bias():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(t, t) == pytest.approx(1.0)
    assert bias(t * 1.1, t) == pytest.approx(0.1)


def test_target_roundtrip():
    rng = np.random.default_rng(0)
    i = rng.choice([-1, 1], 5000) * 10 ** rng.uniform(-16, -3, 5000)
    f = 1e-12
    assert from_target(to_target(i, f), f) == pytest.approx(i, rel=1e-9)
    assert to_target(np.array([0.0]), f)[0] == 0.0


def test_fit_floor():
    i = np.array([1e-9, -2e-9, 3e-9])
    assert fit_floor(i) == pytest.approx(1e-3 * 2e-9)
    assert fit_floor(np.zeros(5)) == 1e-18


def test_features_shape_and_finite():
    shapes = []
    for case in ("NMOS_OFF", "PMOS_ON"):
        df = draw_device_inputs(case, 100, 1)
        kind = case.split("_")[0].lower()
        x = build_features(df, kind)
        assert np.isfinite(x).all()
        shapes.append(x.shape)
    assert shapes[0] == shapes[1] and shapes[0][0] == 100
    # process features are relative to nominal: centred near 0
    assert abs(build_features(draw_device_inputs("NMOS_ON", 4000, 3), "nmos")[:, -5:].mean()) < 0.02
