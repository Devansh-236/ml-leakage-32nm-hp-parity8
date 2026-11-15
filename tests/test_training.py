import inspect

import joblib
import numpy as np
import pytest

from leakml import spec, training

needs_extra_data = pytest.mark.skipif(
    not (spec.ROOT / "data" / "partA_extra" / "NMOS_OFF_1.csv").exists() or not (spec.ROOT / "data" / "partA_extra" / "PMOS_OFF_1.csv").exists(),
    reason="training-only extra data is not shipped in the submission; regenerate with scripts/02_generate_partA.py --extra 8 --edge 4")


def test_split_sizes_disjoint_reproducible():
    a = training.load_splits("NMOS_OFF")
    b = training.load_splits("NMOS_OFF")
    assert (len(a.train), len(a.val), len(a.test)) == spec.SPLIT
    key = lambda d: set(map(tuple, d[["vdd", "temp", "w", "vd", "vs"]].round(12).to_numpy()))  # noqa: E731
    assert not (key(a.train) & key(a.val)) and not (key(a.train) & key(a.test)) and not (key(a.val) & key(a.test))
    assert a.train.equals(b.train) and a.val.equals(b.val) and a.test.equals(b.test)


@needs_extra_data
def test_extra_blocks_train_only():
    base = training.load_splits("PMOS_OFF")
    more = training.load_splits("PMOS_OFF", extra_blocks=2)
    assert len(more.train) == 27000
    assert more.val.equals(base.val) and more.test.equals(base.test)
    assert more.train.iloc[:7000].equals(base.train)


def test_select_best_signature_has_no_test_data():
    assert list(inspect.signature(training.select_best).parameters) == ["candidates", "val"]


class Stub:
    def __init__(self, scale):
        self.scale = scale

    def predict(self, df):
        return df[list(spec.CURRENTS)].to_numpy() * self.scale


def test_select_best_picks_lowest_validation_error():
    val = training.load_splits("NMOS_ON").val
    assert training.select_best({"a": Stub(1.3), "b": Stub(1.01), "c": Stub(0.7)}, val) == "b"


def test_fit_small_and_roundtrip(tmp_path):
    sp = training.load_splits("NMOS_OFF")
    small = training.Splits(sp.train.iloc[:300].reset_index(drop=True), sp.val, sp.test)
    m = training.fit_case_model("NMOS_OFF", "lgbm", small, seed=1)
    ev = training.evaluate(m, sp.val)
    assert set(ev) == set(spec.CURRENTS) and np.isfinite(ev["id"]["median"]) and ev["id"]["median"] < 1.0
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    assert m.predict(sp.val.iloc[:20]) == pytest.approx(m2.predict(sp.val.iloc[:20]), rel=1e-12)


def test_weff_uses_card_wint():
    from leakml.features import weff
    # BSIM4 effective width = W - 2*wint (wint = 5e-9 in both cards), in units of Lmin
    assert weff(np.array([32e-9, 256e-9])) == pytest.approx([22e-9 / 32e-9, 246e-9 / 32e-9])


def test_per_width_target_roundtrip_through_casemodel():
    from leakml.features import fit_floor, to_target, weff
    sp = training.load_splits("NMOS_OFF")
    df = sp.val.iloc[:50]
    cur = df[list(spec.CURRENTS)].to_numpy() / weff(df.w.to_numpy())[:, None]
    floors = [fit_floor(cur[:, j]) for j in range(4)]

    class Oracle:                       # returns exactly the per-width signed-log targets
        def predict(self, X):
            return np.column_stack([to_target(cur[:, j], floors[j]) for j in range(4)])

    m = training.CaseModel("NMOS_OFF", "nmos", [Oracle()], floors, per_width=True)
    assert m.predict(df) == pytest.approx(df[list(spec.CURRENTS)].to_numpy(), rel=1e-9)


def test_region_weights():
    n = training.load_splits("NMOS_OFF").val
    w = training.region_weights("NMOS_OFF", n, 8.0)
    inside = ((n.vs < 0.2 * n.vdd) & (n.vd > 0.7 * n.vdd)).to_numpy()
    assert set(np.unique(w)) <= {1.0, 8.0} and (w[inside] == 8.0).all() and (w[~inside] == 1.0).all()
    p = training.load_splits("PMOS_OFF").val
    wp = training.region_weights("PMOS_OFF", p, 8.0)
    assert (wp[((p.vs > 0.8 * p.vdd) & (p.vd < 0.3 * p.vdd)).to_numpy()] == 8.0).all()
    on = training.load_splits("NMOS_ON").val
    assert (training.region_weights("NMOS_ON", on, 8.0) == 1.0).all()      # ON cases are not reweighted


def test_mlp_accepts_sample_weights():
    from leakml.families import make_family
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (400, 3))
    y = np.column_stack([x[:, 0], x[:, 1]])
    heavy = np.where(x[:, 2] > 0, 50.0, 1.0)
    m = make_family("mlp", seed=0, epochs=60).fit(x, y, sample_weight=heavy)
    assert m.predict(x).shape == (400, 2)


@needs_extra_data
def test_edge_blocks_are_train_only_and_extra_to_uniform_blocks(tmp_path, monkeypatch):
    import pandas as pd
    from leakml import spec as sp_
    base = training.load_splits("NMOS_OFF", extra_blocks=1)
    edge_dir = tmp_path / "partA_edge"
    edge_dir.mkdir()
    fake = pd.read_csv(sp_.ROOT / "data" / "partA" / "NMOS_OFF.csv").head(500)
    for k in (1, 2):
        fake.to_csv(edge_dir / f"NMOS_OFF_{k}.csv", index=False)
    monkeypatch.setattr(training, "EDGE_DIR", edge_dir)
    both = training.load_splits("NMOS_OFF", extra_blocks=1, edge_blocks=2)
    assert len(both.train) == len(base.train) + 1000
    assert both.val.equals(base.val) and both.test.equals(base.test)
    assert both.train.iloc[:len(base.train)].equals(base.train)
