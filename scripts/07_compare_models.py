"""Part A: four model families x four cases, trained on the 7,000-row training split,
scored on the 1,500-row validation split (deliverable 4)."""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.families import FAMILIES  # noqa: E402
from leakml.reporting import comparison_markdown, eval_rows  # noqa: E402
from leakml.training import fit_case_model, load_splits  # noqa: E402


def main():
    res = spec.ROOT / "data" / "partA_results"
    (res / "L0").mkdir(parents=True, exist_ok=True)
    (spec.ROOT / "results" / "partA").mkdir(parents=True, exist_ok=True)
    rows = []
    for case in spec.CASES:
        sp = load_splits(case)
        for fam in FAMILIES:
            path = res / "L0" / f"{case}_{fam}.joblib"
            if path.exists():
                model = joblib.load(path)
            else:
                model = fit_case_model(case, fam, sp, seed=spec.SEEDS["models"])
                joblib.dump(model, path)
            rows += eval_rows(case, fam, model, sp.val, "val")
            r = [x for x in rows[-4:]]
            print(case, fam, " ".join(f"{x['terminal']}={100 * x['median']:.3g}%" for x in r), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(res / "compare_val.csv", index=False)
    (spec.ROOT / "results" / "partA" / "COMPARISON.md").write_text(
        comparison_markdown(df, "Model comparison on the validation set (7,000 training rows)") + "\n")


if __name__ == "__main__":
    main()
