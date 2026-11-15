"""Every deliverable of the PDF (section 6) exists and has the right shape."""
import joblib
import pandas as pd
import pytest

from leakml import spec

R = spec.ROOT
CASES = list(spec.CASES)
TEX = R / "results" / "REPORT.tex"   # LaTeX source is not part of the submission
needs_tex = pytest.mark.skipif(not TEX.exists(), reason="LaTeX source not shipped in the submission (REPORT.pdf is)")


@pytest.mark.parametrize("case", CASES)
def test_dataset_csv(case):
    df = pd.read_csv(R / "data" / "partA" / f"{case}.csv")
    assert len(df) == 10000 and list(df.columns) == list(spec.DATA_COLS) and not df.isna().any().any()


def test_example_netlists():
    names = [f"partA/{c}.cir" for c in CASES] + [f"partB/{g}.cir" for g in ("INV", "NAND2", "NOR2", "AND2", "OR2")] + ["partC/parity8.cir"]
    for n in names:
        text = (R / "netlists" / n).read_text()
        assert spec.OPTIONS in text, n


def test_data_check_and_trends():
    assert "Kirchhoff" in (R / "results/partA/DATA_CHECK.md").read_text()
    for n in ("trend_off_vs_temp", "trend_off_vs_vth0", "trend_gate_vs_toxe"):
        assert (R / "results/partA" / f"{n}.png").stat().st_size > 5000


def test_model_comparison_table():
    df = pd.read_csv(R / "data/partA_results/compare_val.csv")
    assert df.groupby(["case", "family"]).ngroups == 16 and {"r2", "median", "p95"} <= set(df.columns)


def test_chosen_models_load_and_predict():
    ch = pd.read_csv(R / "data/partA_results/chosen.csv")
    assert set(ch.case) == set(CASES) and {"median", "p95"} <= set(ch.columns)
    for case in CASES:
        m = joblib.load(R / "trained_models" / f"{case}.joblib")
        df = pd.read_csv(R / "data" / "partA" / f"{case}.csv").head(5)
        assert m.predict(df).shape == (5, 4)
        assert (R / "results/partA" / f"pred_vs_spice_{case}.png").exists()


def test_gate_results():
    t = pd.read_csv(R / "data/partB/gates_table.csv", dtype={"pattern": str})
    a = t[t.variant == "a"]
    assert len(a) == 18 and {"spice_median", "ml_median", "bias", "median_err", "p95_err"} <= set(a.columns)
    assert (R / "results/partB/gates_parity.png").exists() and (R / "results/partB/GATES.md").exists()
    ml = pd.read_csv(R / "data/partB/gates_ml.csv")
    assert {"spice_leak", "ml_leak", "spice_x", "ml_x", "spice_out"} <= set(ml.columns)
    assert len(ml[ml.variant == "a"]) == 3600


def test_parity_results():
    p = pd.read_csv(R / "data/partC/parity_ml.csv")
    assert len(p) == 51200 and {"spice_leak", "ml_leak", "err"} <= set(p.columns)
    assert (R / "results/partC/parity_parity.png").exists() and (R / "results/partC/PARITY.md").exists()


def test_report_and_docs():
    assert (R / "Report.pdf").stat().st_size > 50_000
    for n in ("README.md", "DELIVERABLES.md", "scripts/run_all.sh"):
        assert (R / n).exists(), n
    if not TEX.exists():                  # the submission ships the PDF only, not the LaTeX source
        return
    tex = TEX.read_text()
    for section in ("Setup", "Data generation", "Models", "Gate results", "Test-circuit results",
                    "Error analysis", "Limitations and improvements"):
        assert section in tex, section


@needs_tex
def test_report_has_no_unfinished_markers():
    tex = TEX.read_text()
    assert "TODO-FINAL" not in tex and "??" not in tex


@needs_tex
def test_report_shows_per_case_data_check_and_error_figure():
    tex = TEX.read_text()
    assert "tables/datacheck.tex" in tex and (R / "results/tables/datacheck.tex").exists()
    assert "error_vs_corner.png" in tex and "REGION\\_DIAGNOSTIC" in tex   # underscores are escaped in the .tex source
    assert (R / "results/partA/REGION_DIAGNOSTIC.md").exists()
