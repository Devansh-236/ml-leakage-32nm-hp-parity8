"""Four model families behind one API.

Every family maps a feature matrix X (n, k) to a target matrix T (n, 4) (signed-log
currents id, ig, is, ib, standardised internally where it helps) and implements
fit(X, T) / predict(X) -> (n, 4).
"""
import numpy as np
import torch  # noqa: F401  must be imported before lightgbm: both bundle OpenMP and the reverse order segfaults on macOS

FAMILIES = ("poly", "lgbm", "mlp", "gp")


class Poly:
    """Standardise -> polynomial features -> ridge. Degree (3 or 4) is picked on a holdout."""

    def __init__(self, seed=0, degrees=(3, 4), **_):
        self.seed, self.degrees = seed, degrees

    def _make(self, degree):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                             StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 8)))

    def fit(self, X, T, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        tr, ho = idx[:cut], idx[cut:]
        best, best_err = None, np.inf
        for d in self.degrees:
            err = np.mean((self._make(d).fit(X[tr], T[tr]).predict(X[ho]) - T[ho]) ** 2)
            if err < best_err:
                best, best_err = d, err
        self.degree = best
        self.model = self._make(best).fit(X, T, **self._w(sample_weight))
        return self

    @staticmethod
    def _w(sample_weight):
        return {} if sample_weight is None else {"ridgecv__sample_weight": sample_weight}

    def predict(self, X):
        return self.model.predict(X)


class LGBM:
    """Gradient-boosted trees, one regressor per terminal, early stopping on 10 % of the rows."""

    def __init__(self, seed=0, n_estimators=3000, learning_rate=0.05, num_leaves=63, n_jobs=1, **_):
        self.seed, self.n, self.lr, self.leaves, self.n_jobs = seed, n_estimators, learning_rate, num_leaves, n_jobs

    def fit(self, X, T, sample_weight=None):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        cut = int(0.9 * len(X))
        tr, ho = idx[:cut], idx[cut:]
        self.models = []
        for j in range(T.shape[1]):
            m = lgb.LGBMRegressor(n_estimators=self.n, learning_rate=self.lr, num_leaves=self.leaves,
                                  min_child_samples=5, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                                  random_state=self.seed, verbose=-1, n_jobs=self.n_jobs)
            m.fit(X[tr], T[tr, j], sample_weight=None if sample_weight is None else sample_weight[tr],
                  eval_set=[(X[ho], T[ho, j])], callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models.append(m)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


class MLP:
    """Fully connected network (SiLU), multi-output on standardised targets, Adam + cosine decay."""

    def __init__(self, seed=0, width=128, depth=3, epochs=300, lr=3e-3, batch=256, weight_decay=0.0, **_):
        self.seed, self.width, self.depth = seed, width, depth
        self.epochs, self.lr, self.batch, self.wd = epochs, lr, batch, weight_decay

    def _net(self, k_in, k_out):
        import torch.nn as nn
        layers, d = [], k_in
        for _ in range(self.depth):
            layers += [nn.Linear(d, self.width), nn.SiLU()]
            d = self.width
        return nn.Sequential(*layers, nn.Linear(d, k_out))

    def fit(self, X, T, sample_weight=None):
        import torch
        torch.manual_seed(self.seed)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        self.tm, self.ts = T.mean(0), T.std(0) + 1e-12
        x = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        y = torch.tensor((T - self.tm) / self.ts, dtype=torch.float32)
        wts = torch.ones(len(x)) if sample_weight is None else torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
        self.net = self._net(x.shape[1], y.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(x) / self.batch))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(x), generator=gen)
            for i in range(0, len(x), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = (wts[b, None] * (self.net(x[b]) - y[b]) ** 2).mean()
                loss.backward()
                opt.step()
                sched.step()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            out = self.net(torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)).numpy()
        return out * self.ts + self.tm


class GP:
    """Gaussian process (ARD RBF + noise), one per terminal. Kernel hyperparameters are
    optimised on a subset, the final fit uses all rows with the kernel fixed."""

    def __init__(self, seed=0, subset=1500, **_):
        self.seed, self.subset = seed, subset

    def fit(self, X, T, sample_weight=None):          # sample weights are ignored by the GP
        from sklearn.gaussian_process import GaussianProcessRegressor as GPR
        from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.xm) / self.xs
        rng = np.random.default_rng(self.seed)
        sub = rng.permutation(len(Z))[:self.subset]
        self.models = []
        for j in range(T.shape[1]):
            kern = C(1.0, (1e-3, 1e3)) * RBF(np.full(Z.shape[1], 3.0), (1e-2, 1e3)) + WhiteKernel(1e-4, (1e-9, 1e-1))
            g = GPR(kern, normalize_y=True, random_state=self.seed).fit(Z[sub], T[sub, j])
            full = GPR(g.kernel_, normalize_y=True, optimizer=None).fit(Z, T[:, j])
            full.L_ = None      # only needed for predictive std; dropping it keeps saved models small
            self.models.append(full)
        return self

    def predict(self, X):
        Z = (X - self.xm) / self.xs
        return np.column_stack([m.predict(Z) for m in self.models])


_REGISTRY = {"poly": Poly, "lgbm": LGBM, "mlp": MLP, "gp": GP}


def make_family(name: str, seed: int, **kw):
    return _REGISTRY[name](seed=seed, **kw)
