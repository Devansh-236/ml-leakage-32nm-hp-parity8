#!/bin/sh
# Regenerate everything, in order, with the fixed seeds in leakml/spec.py.
# Needs ngspice (brew install ngspice), the venv from requirements.txt. The report is Report.pdf at the top level.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/01_verify.py                    # simulator and nominal operating points
$PY scripts/02_generate_partA.py --extra 8 --edge 4  # four 10,000-row datasets + training-only blocks (8 uniform, 4 rail-concentrated)
$PY scripts/03_check_data.py                # Kirchhoff check and trend plots
$PY scripts/04_example_netlists.py          # example netlists
$PY scripts/05_gates_spice.py               # Part B SPICE: reported and development corners
$PY scripts/06_parity_spice.py              # Part C SPICE: 51,200 runs
$PY scripts/07_compare_models.py            # level 0: 4 families x 4 cases
$PY scripts/08_select_models.py             # level 0: best family per case (writes trained_models/ for the ladder to start from)
$PY scripts/11_run_ladder.py --workers 10   # escalation ladder (several hours), overwrites trained_models/ with the best rung
$PY scripts/08_select_models.py --final     # score the final models once on the test split (chosen.csv, MODELS.md)
$PY scripts/09_gates_ml.py report           # Part B ML vs SPICE
$PY scripts/10_parity_ml.py                 # Part C ML vs SPICE
$PY scripts/12_error_analysis.py            # error analysis and leakage components
$PY scripts/14_region_diagnostic.py        # transistor-level error in the gate-relevant region (fresh SPICE rows, evaluation only)
$PY scripts/18_nand_parity.py              # second XOR2 topology (4 x NAND2): SPICE + ML at the reported corners
$PY scripts/17_fresh_check.py              # 200 fresh corners: new SPICE references, final models unchanged
