import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leakml import ladder, spec, training


class FastFamily:
    """Stand-in for an MLP: a seeded ridge fit, fast and deterministic."""

    def __init__(self, seed=0, **kw):
        self.seed = seed

    def fit(self, X, T, sample_weight=None):
        from sklearn.linear_model import Ridge
        self.m = Ridge(alpha=1e-3).fit(X, T, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.m.predict(X)


_REAL_LOAD = training.load_splits


def small_splits(case, extra_blocks=0):
    sp = _REAL_LOAD(case, 0)
    return training.Splits(sp.train.iloc[:400].reset_index(drop=True), sp.val.iloc[:200].reset_index(drop=True), sp.test)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(ladder, "load_splits", lambda case, extra_blocks=0, edge_blocks=0: small_splits(case))
    monkeypatch.setattr(training, "make_family", lambda name, seed, **kw: FastFamily(seed=seed))


def test_job_paths_and_variants():
    assert set(ladder.VARIANTS) >= {"k1", "k2", "k4", "k8", "plain", "long"}
    assert ladder.VARIANTS["k8"]["blocks"] == 8 and ladder.VARIANTS["plain"]["per_width"] is False
    j = ladder.Job("L1", "NMOS_OFF", "k2", 1)
    assert ladder.job_path(Path("/w"), j) == Path("/w/L1/NMOS_OFF__k2__s1.joblib")


def test_edge_variant_and_level3_jobs():
    v = ladder.variant_spec("k8e4")
    assert v["blocks"] == 8 and v["edge"] == 4 and v["per_width"] is True
    assert ladder.variant_spec("k8").get("edge", 0) == 0
    jobs = ladder.jobs_for("L3", {c: "k8" for c in spec.CASES})
    assert len(jobs) == 12 and {j.variant for j in jobs} == {"k8e4"} and {j.seed for j in jobs} == {1, 2, 3}
    assert {j.rung for j in jobs} == {"L3"}


def test_run_job_passes_edge_blocks(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ladder, "load_splits", lambda case, extra_blocks=0, edge_blocks=0:
                        seen.update(extra=extra_blocks, edge=edge_blocks) or small_splits(case))
    ladder.run_job(ladder.Job("L3", "NMOS_OFF", "k8e4", 1), tmp_path)
    assert seen == {"extra": 8, "edge": 4}


def test_run_job_resumes(tmp_path):
    j = ladder.Job("L1", "NMOS_OFF", "k1", 1)
    p = ladder.run_job(j, tmp_path)
    m1 = os.stat(p).st_mtime_ns
    df = small_splits("NMOS_OFF").val
    import joblib
    a = joblib.load(p).predict(df)
    p2 = ladder.run_job(j, tmp_path)
    assert p2 == p and os.stat(p).st_mtime_ns == m1
    assert joblib.load(p).predict(df) == pytest.approx(a, rel=1e-12)
    assert p.with_suffix(".json").exists()


def test_partial_write_ignored(tmp_path):
    j = ladder.Job("L1", "PMOS_ON", "k1", 1)
    p = ladder.job_path(tmp_path, j)
    p.parent.mkdir(parents=True)
    Path(str(p) + ".tmp").write_bytes(b"garbage from an interrupted write")
    out = ladder.run_job(j, tmp_path)
    assert out == p and p.exists()
    import joblib
    assert joblib.load(p).predict(small_splits("PMOS_ON").val).shape == (200, 4)


def test_run_job_never_touches_test_split(tmp_path, monkeypatch):
    def guarded(case, extra_blocks=0, edge_blocks=0):
        sp = small_splits(case)

        class NoTest(training.Splits):
            @property
            def test(self):
                raise AssertionError("ladder touched the test split")

            @test.setter
            def test(self, v):
                pass
        return NoTest(sp.train, sp.val, None)
    monkeypatch.setattr(ladder, "load_splits", guarded)
    ladder.run_job(ladder.Job("L1", "NMOS_ON", "k1", 1), tmp_path)


def test_assemble_averages_members(tmp_path):
    for s in (1, 2, 3):
        ladder.run_job(ladder.Job("L2", "NMOS_OFF", "k1", s), tmp_path)
    m = ladder.assemble("NMOS_OFF", [ladder.job_path(tmp_path, ladder.Job("L2", "NMOS_OFF", "k1", s)) for s in (1, 2, 3)])
    assert len(m.members) == 3 and m.predict(small_splits("NMOS_OFF").val).shape == (200, 4)


def test_pick_best_prefers_fewer_rows_on_tie():
    cand = {"k8": {"score": 0.5, "rows": 87000}, "k2": {"score": 0.5, "rows": 27000}, "k4": {"score": 0.6, "rows": 47000}}
    assert ladder.pick_best(cand) == "k2"


def test_dev_accuracy_reads_only_dev_files(monkeypatch):
    seen = []
    real = pd.read_csv

    def spy(path, *a, **k):
        seen.append(str(path))
        if "report" in str(path):
            raise AssertionError("ladder read a reported-corner file")
        return real(path, *a, **k)
    monkeypatch.setattr(pd, "read_csv", spy)
    from leakml.oracle import OracleModels
    acc = ladder.dev_accuracy(OracleModels(), n_corners=2, iters=12)
    assert set(acc) == {"a", "b"} and all(0.9 < v <= 1.0 for v in acc.values())
    assert all("dev" in s for s in seen)
