import numpy as np

from leakml import devices, spec


def test_generate_case_small():
    df, failed = devices.generate_case("NMOS_OFF", 40, seed=1, workers=2)
    assert len(df) == 40 and failed == 0
    assert list(df.columns) == list(spec.DATA_COLS)
    assert not df.isna().any().any()
    cur = df[list(spec.CURRENTS)].to_numpy()
    resid = np.abs(cur.sum(axis=1)) / np.abs(cur).max(axis=1)
    assert (resid < 1e-6).all()


def test_deterministic_across_workers(tmp_path):
    a, _ = devices.generate_case("PMOS_ON", 30, seed=5, workers=1)
    b, _ = devices.generate_case("PMOS_ON", 30, seed=5, workers=4)
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_failed_samples_redrawn(monkeypatch):
    real = devices._simulate
    calls = {"n": 0}

    def flaky(kind, df, workers=None):
        out = real(kind, df, workers)
        if calls["n"] == 0:
            out[:3] = [None, None, None]
        calls["n"] += 1
        return out

    monkeypatch.setattr(devices, "_simulate", flaky)
    df, failed = devices.generate_case("NMOS_ON", 40, seed=1, workers=2)
    assert len(df) == 40 and failed == 3 and not df.isna().any().any()


def test_generate_case_edge_rows_are_simulated_cleanly():
    df, failed = devices.generate_case("NMOS_OFF", 40, seed=2, workers=2, edge_fraction=1.0)
    assert len(df) == 40 and not df.isna().any().any()
    on_rail = ((df.vd == 0) | (df.vd == df.vdd)).sum()
    assert on_rail >= 10
    cur = df[list(spec.CURRENTS)].to_numpy()
    assert (np.abs(cur.sum(axis=1)) <= 1e-6 * np.abs(cur).max(axis=1) + 2e-15).all()
