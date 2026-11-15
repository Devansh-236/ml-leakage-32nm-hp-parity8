"""Escalation ladder for the transistor models (see leakml/ladder.py).

Decision rule, fixed before any run: a rung passes when the gate-level mean accuracy on the
DEVELOPMENT corners is >= TARGET_ACCURACY for both variants (a: SPICE node voltages, b: model-solved
nodes). The ladder stops at the first passing rung (use --full to run every rung anyway). The best
rung's models go to trained_models/. Resumable: finished jobs are never retrained.
Run as:  caffeinate -i .venv/bin/python scripts/11_run_ladder.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import ladder, spec  # noqa: E402
from leakml.ladder import Job, job_path  # noqa: E402
from leakml.training import evaluate, load_splits  # noqa: E402

TARGET_ACCURACY = 0.99


def score(model, case):
    return ladder.mean_median_error(evaluate(model, load_splits(case).val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run every rung even after one passes")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    res = spec.ROOT / "data" / "partA_results"
    work = res / "ladder"
    log, rung_models, rung_acc = [], {}, {}

    def record(rung, models, variants, members):
        acc = ladder.dev_accuracy(models)
        rung_models[rung], rung_acc[rung] = models, acc
        for case in spec.CASES:
            log.append({"rung": rung, "case": case, "variant": variants[case], "members": members[case],
                        "val_mean_median_err": score(models[case], case), "dev_acc_a": acc["a"], "dev_acc_b": acc["b"]})
        pd.DataFrame(log).to_csv(res / "ladder_log.csv", index=False)
        print(f"{rung}: dev-corner accuracy a={100 * acc['a']:.2f}%  b={100 * acc['b']:.2f}%", flush=True)
        return min(acc.values()) >= TARGET_ACCURACY

    # L0: the models chosen by scripts 07/08 (all families, 7,000 rows)
    l0 = {c: joblib.load(spec.ROOT / "trained_models" / f"{c}.joblib") for c in spec.CASES}
    passed = record("L0", l0, {c: l0[c].info["family"] for c in l0}, {c: 1 for c in l0})

    best, sizes = {}, []
    if not passed or args.full:
        # L1: the accuracy-versus-rows curve
        ladder.run_jobs(ladder.jobs_for("L1"), work, args.workers)
        models, rows = {}, []
        for case in spec.CASES:
            cands = {}
            for size in ladder.LADDER_SIZES:
                p = job_path(work, Job("L1", case, size, 1))
                meta = pd.read_json(p.with_suffix(".json"), typ="series")
                cands[size] = {"score": meta["score"], "rows": meta["rows"], "path": p}
                rows.append({"case": case, "variant": size, "rows": meta["rows"], "val_mean_median_err": meta["score"]})
            best[case] = ladder.pick_best(cands)
            models[case] = ladder.assemble(case, [cands[best[case]]["path"]])
        pd.DataFrame(rows).to_csv(res / "data_size.csv", index=False)
        passed = record("L1", models, best, {c: 1 for c in models})

    if (not passed or args.full) and best:
        # L2: three-seed average at the best size
        ladder.run_jobs(ladder.jobs_for("L2", best), work, args.workers)
        models = {c: ladder.assemble(c, [job_path(work, Job("L1", c, best[c], 1))] +
                                     [job_path(work, Job("L2", c, best[c], s)) for s in (2, 3)]) for c in spec.CASES}
        passed = record("L2", models, best, {c: 3 for c in models})

    if (not passed or args.full) and best:
        # L3: edge-augmented training (40,000 extra rows at the rails), three seeds averaged
        ladder.run_jobs(ladder.jobs_for("L3", best), work, args.workers)
        models = {c: ladder.assemble(c, [job_path(work, Job("L3", c, "k8e4", s)) for s in (1, 2, 3)]) for c in spec.CASES}
        passed = record("L3", models, {c: "k8e4" for c in models}, {c: 3 for c in models})

    # keep the best rung (highest of min(a, b); the earlier rung on a tie)
    best_rung = max(rung_acc, key=lambda r: (round(min(rung_acc[r].values()), 6), -list(rung_acc).index(r)))
    print("best rung:", best_rung, {k: f"{100 * v:.2f}%" for k, v in rung_acc[best_rung].items()})
    (res / "final").mkdir(exist_ok=True)
    for c, m in rung_models[best_rung].items():
        joblib.dump(m, res / "final" / f"{c}.joblib")
        shutil.copy(res / "final" / f"{c}.joblib", spec.ROOT / "trained_models" / f"{c}.joblib")
    (res / "final" / "RUNG.txt").write_text(best_rung + "\n")


if __name__ == "__main__":
    main()
