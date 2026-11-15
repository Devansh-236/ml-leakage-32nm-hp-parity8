# ML-based leakage estimation, 32 nm HP, VDD = 1 V, 8-bit parity checker

Devansh Varshney (2023102042), Vansh Agarwal (2023102043), Dhruv Agarwal (2023102048), Datarnoor Singh Oberoi (2023102047)

Digital VLSI Design course project. We train four machine-learning models on SPICE data for single NMOS and
PMOS transistors (ON and OFF), build the leakage of INV, NAND2, NOR2, AND2 and OR2 from them, and then the
leakage of an 8-bit parity checker from the gates. Everything is compared against transistor-level ngspice
runs at 200 random PVT corners.

Mean accuracy on the 200 test corners:

| | accuracy |
|---|---|
| gates, internal nodes taken from SPICE | 99.21 % |
| gates, internal nodes solved with our models | 99.13 % |
| 8-bit parity checker (51,200 runs) | 99.29 % |

On 200 fresh corners, drawn after the models were final, the same models give 99.21 %, 99.08 % and 99.35 %; with every XOR2 built from four NAND2 the parity checker gives 99.20 %.

The report is `Report.pdf`. `DELIVERABLES.md` lists where each item asked for in the project text is.

## Setup

You need ngspice 46 (`brew install ngspice`), Python 3.11 and the packages in `requirements.txt`:

    python3.11 -m venv .venv
    .venv/bin/pip install -r requirements.txt

The model file is `models/32nm_HP.pm`, as supplied.

## Running

    sh scripts/run_all.sh          # every step in order, seeds fixed in leakml/spec.py
    .venv/bin/python -m pytest     # tests

The training step (`scripts/11_run_ladder.py`) takes several hours on a 10-core Mac. It can be interrupted and
restarted. Data generation and the SPICE reference runs take a few tens of minutes.

## Folders

- `data/partA/`: the four 10,000-row transistor datasets
- `data/partB/`, `data/partC/`: SPICE and ML results for the gates and the parity checker
- `data/fresh/`: the 200 fresh corners and the summary of the fresh-corner check
- `data/partA_results/`: model comparison, chosen-model scores, training log
- `trained_models/`: the four saved models
- `netlists/`: example netlists
- `results/`: tables and plots for each part
- `leakml/`: the code library; `scripts/`: one script per step; `tests/`: tests

The extra training data (`data/partA_extra/`, `data/partA_edge/`) and the intermediate model files are not
included. `scripts/run_all.sh` regenerates them from the seeds. The per-run results of the two extra checks
(`data/fresh/*_spice_fresh.csv`, `*_ml_fresh.csv`, `data/partC/parity_nand_spice.csv`, `parity_nand_ml.csv`)
are left out of the zip archive to keep it small. They are in the GitHub repository
(https://github.com/Devansh-236/ml-leakage-32nm-hp-parity8), `scripts/17_fresh_check.py` and
`scripts/18_nand_parity.py` regenerate them, and their summaries are in `results/FRESH_CHECK.md` and
`results/partC/PARITY_NAND.md`.

## Notes

- The final models train on 127,000 rows per case: the 7,000 training rows, 80,000 extra uniform rows and 40,000
  rows with Vd and Vs close to the supply rails. Validation and test rows only come from the 10,000-row datasets.
  Trained on the 7,000 rows alone, the models reach about 86 % at gate level. The report explains why.
- Leakage is the supply current plus the current delivered by inputs held at VDD. Each XOR2 is built as
  AND2(OR2(a, b), NAND2(a, b)).
- The training rungs were accepted or rejected using 200 separate development corners. The test split and the
  reported corners are only scored by the final scripts.
