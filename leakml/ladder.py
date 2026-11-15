"""The escalation ladder for the transistor models.

L0  7,000 training rows, four families (scripts 07/08).
L1  one MLP trained on 7,000 + k x 10,000 rows, k = 1, 2, 4, 8 (the accuracy-versus-rows curve).
L2  average of three MLPs (seeds) at the best L1 size.
L3  edge-augmented training: the L1/L2 size plus 4 blocks (40,000 rows) of training-only samples with
    Vd/Vs concentrated at the rails, three seeds averaged. Added after a SPICE check showed the L2 error
    sits at the exact rail voltages the gates use (see the report). Plain-target and
    2x-epoch variants remain defined (VARIANTS "plain", "long") but are not run.

A job is one (case, variant, seed) training run. Jobs are resumable: results are written
atomically and a finished job is never retrained. Ladder decisions use validation data and the
development corners only; the test split and the reported corners are not read here.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from . import spec
from .training import CaseModel, evaluate, fit_case_model, load_splits

MLP_KW = dict(width=256, depth=4, epochs=1500, lr=1e-3)
VARIANTS = {
    **{f"k{k}": dict(blocks=k, per_width=True, region_weight=8.0, kw=MLP_KW) for k in (1, 2, 4, 8)},
    "k8e4": dict(blocks=8, edge=4, per_width=True, region_weight=8.0, kw=MLP_KW),
    "plain": dict(blocks=None, per_width=False, region_weight=8.0, kw=MLP_KW),
    "long": dict(blocks=None, per_width=True, region_weight=8.0, kw={**MLP_KW, "epochs": 2 * MLP_KW["epochs"]}),
}
LADDER_SIZES = ("k1", "k2", "k4", "k8")


@dataclass(frozen=True)
class Job:
    rung: str
    case: str
    variant: str        # "k4", or "<template>@k4" for plain / long
    seed: int


def variant_spec(name: str) -> dict:
    if "@" in name:
        base, size = name.split("@")
        return {**VARIANTS[base], "blocks": VARIANTS[size]["blocks"]}
    return VARIANTS[name]


def job_path(workdir: Path, job: Job) -> Path:
    return Path(workdir) / job.rung / f"{job.case}__{job.variant}__s{job.seed}.joblib"


def mean_median_error(metrics: Mapping) -> float:
    return float(np.mean([m["median"] for m in metrics.values()]))


def run_job(job: Job, workdir: Path) -> Path:
    """Train one model (or return the finished one). Validation data only."""
    path = job_path(workdir, job)
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return path
    v = variant_spec(job.variant)
    sp = load_splits(job.case, v["blocks"], v.get("edge", 0))
    model = fit_case_model(job.case, "mlp", sp, seed=spec.SEEDS["models"] + job.seed,
                           per_width=v["per_width"], region_weight=v["region_weight"], **v["kw"])
    metrics = evaluate(model, sp.val)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"job": job.__dict__, "rows": len(sp.train), "score": mean_median_error(metrics),
                                "val": metrics}, indent=1))
    tmp = Path(str(path) + ".tmp")
    joblib.dump(model, tmp)
    os.replace(tmp, path)                       # atomic: the model file only exists once complete
    return path


def _run(args):
    return run_job(*args)


def run_jobs(jobs: Sequence[Job], workdir: Path, workers: int = 10) -> list:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, [(j, workdir) for j in jobs]))


def jobs_for(rung: str, best: Mapping[str, str] = None) -> list:
    """Jobs of a rung. `best` maps case -> the size variant ("k4") chosen by L1."""
    out = []
    for case in spec.CASES:
        if rung == "L1":
            out += [Job("L1", case, size, 1) for size in LADDER_SIZES]
        elif rung == "L2":
            out += [Job("L2", case, best[case], s) for s in (2, 3)]
        elif rung == "L3":
            out += [Job("L3", case, "k8e4", s) for s in (1, 2, 3)]
        else:
            raise KeyError(rung)
    return out


def assemble(case: str, paths: Sequence[Path]) -> CaseModel:
    """One CaseModel averaging the members of several trained models (same case and targets)."""
    models = [joblib.load(p) for p in paths]
    first = models[0]
    for m in models[1:]:
        if m.per_width != first.per_width or not np.allclose(m.floors, first.floors):
            raise ValueError(f"{case}: members were trained with different targets")
    members = [mem for m in models for mem in m.members]
    return CaseModel(case, first.kind, members, first.floors,
                     {**first.info, "members": len(members)}, per_width=first.per_width)


def pick_best(candidates: Mapping[str, Mapping]) -> str:
    """Lowest score wins; on a tie the variant with fewer training rows."""
    return min(candidates, key=lambda k: (round(candidates[k]["score"], 9), candidates[k]["rows"]))


def dev_accuracy(models: Mapping, n_corners: int = None, iters: int = 30) -> dict:
    """Pooled mean accuracy of the gate leakage on the development corners (never the reported
    ones): {"a": with SPICE node voltages, "b": with model-solved nodes}."""
    from .evaluate import gate_predictions, pooled
    d = spec.ROOT / "data" / "partB"
    corners = pd.read_csv(d / "corners_dev.csv")
    sp = pd.read_csv(d / "gates_spice_dev.csv", dtype={"pattern": str})
    if n_corners:
        corners = corners.iloc[:n_corners]
        sp = sp[sp.corner_id.isin(corners.corner_id)]
    return {v: pooled(gate_predictions(models, corners, sp, v, iters=iters))["accuracy"] for v in ("a", "b")}
