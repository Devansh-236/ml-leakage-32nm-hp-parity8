import numpy as np
import pytest

from leakml.families import FAMILIES, make_family


def toy(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 8))
    y = np.column_stack([0.5 * x[:, 0] * x[:, 1] + np.sin(x[:, 2] + j * x[:, 3]) + 0.3 * x[:, 4] ** 2 - x[:, 5]
                         for j in range(4)])
    return x, y


def r2(p, t):
    return 1 - ((t - p) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)


@pytest.mark.parametrize("name", FAMILIES)
def test_family_fits_smooth_function(name):
    xt, yt = toy(1500, 0)
    xe, ye = toy(300, 1)
    fam = make_family(name, seed=0)
    fam.fit(xt, yt)
    p = fam.predict(xe)
    assert p.shape == (300, 4) and np.isfinite(p).all()
    assert (r2(p, ye) > 0.9).all(), r2(p, ye)


def test_mlp_seed_reproducible():
    xt, yt = toy(400, 0)
    xe, _ = toy(50, 1)
    a, b = make_family("mlp", seed=3, epochs=30), make_family("mlp", seed=3, epochs=30)
    a.fit(xt, yt)
    b.fit(xt, yt)
    assert a.predict(xe) == pytest.approx(b.predict(xe), abs=1e-6)


def test_unknown_family():
    with pytest.raises(KeyError):
        make_family("nope", seed=0)


def test_lgbm_then_mlp_in_one_process():
    """torch and LightGBM ship separate OpenMP runtimes; on macOS fitting LightGBM and then an
    MLP in one process used to segfault. Run it in a subprocess so a crash fails the test only."""
    import subprocess
    import sys
    code = (
        "import numpy as np\n"
        "from leakml.families import make_family\n"
        "rng = np.random.default_rng(0); x = rng.uniform(-1, 1, (300, 8)); y = np.sin(x[:, :4])\n"
        "make_family('lgbm', seed=0).fit(x, y)\n"
        "make_family('mlp', seed=0, epochs=5).fit(x, y)\n"
        "make_family('lgbm', seed=0).fit(x, y)\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-300:]


def test_gp_pickle_is_small_and_predicts_identically():
    import pickle
    xt, yt = toy(600, 0)
    xe, _ = toy(40, 1)
    fam = make_family("gp", seed=0, subset=300)
    fam.fit(xt, yt)
    before = fam.predict(xe)
    blob = pickle.dumps(fam)
    assert len(blob) < 1_000_000, len(blob)          # no n x n Cholesky factors in the saved model
    assert pickle.loads(blob).predict(xe) == pytest.approx(before, rel=1e-9)


def test_saved_lgbm_model_loads_after_torch_import(tmp_path):
    """Unpickling a LightGBM model in a process that already imported torch used to segfault
    (two OpenMP runtimes). `import leakml` must make that safe."""
    import subprocess
    import sys
    pkl = tmp_path / "m.pkl"
    make = ("import pickle, numpy as np, lightgbm as lgb\n"
            "x = np.random.default_rng(0).uniform(-1, 1, (300, 5))\n"
            f"pickle.dump(lgb.LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1).fit(x, x[:, 0]), open(r'{pkl}', 'wb'))\n")
    assert subprocess.run([sys.executable, "-W", "ignore", "-c", make], timeout=120).returncode == 0
    use = ("import leakml, torch, pickle, numpy as np\n"
           f"m = pickle.load(open(r'{pkl}', 'rb'))\n"
           "net = torch.nn.Linear(5, 1); net(torch.zeros(3, 5)).sum().backward()\n"
           f"m = pickle.load(open(r'{pkl}', 'rb')); print('ok')\n")
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", use], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, (r.returncode, r.stderr[-300:])
