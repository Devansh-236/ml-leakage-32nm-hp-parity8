# Where to find each deliverable

Team: Devansh Varshney (2023102042), Vansh Agarwal (2023102043), Dhruv Agarwal (2023102048), Datarnoor Singh Oberoi (2023102047)

| # | Deliverable | File |
|---|---|---|
| 1 | Datasets (four CSVs, 10,000 rows each) | `data/partA/NMOS_ON.csv`, `NMOS_OFF.csv`, `PMOS_ON.csv`, `PMOS_OFF.csv` |
| 2 | Example netlists | `netlists/partA/` (one per case), `netlists/partB/` (one per gate), `netlists/partC/parity8.cir` |
| 3 | Data check and trend plots | `results/partA/DATA_CHECK.md`, `results/partA/trend_*.png` |
| 4 | Model comparison table | `results/partA/COMPARISON.md`, `data/partA_results/compare_val.csv`, Table 2 and Appendix A of the report |
| 5 | Chosen models | `results/partA/MODELS.md`, `data/partA_results/chosen.csv`, `results/partA/pred_vs_spice_*.png`, `trained_models/*.joblib` |
| 6 | Gate results table | `results/partB/GATES.md`, `data/partB/gates_table.csv` |
| 7 | Gate parity plot | `results/partB/gates_parity.png` |
| 8 | Test-circuit results and parity plot | `results/partC/PARITY.md`, `data/partC/parity_table.csv`, `results/partC/parity_parity.png`; second XOR2 topology (4 NAND2) in `results/partC/PARITY_NAND.md` |
| 9 | Code | `scripts/run_all.sh`, `scripts/01_*.py` to `scripts/12_*.py`, `14_*.py`, `17_*.py` and `18_*.py`, `leakml/`, `tests/` |
| 10 | Report | `Report.pdf` |

Also included: `results/FRESH_CHECK.md` (the final models on 200 fresh corners, with new SPICE references), `results/ERROR_ANALYSIS.md` and `results/error_vs_corner.png` (error against corner parameters),
`results/partB/LEAKAGE_COMPONENTS.md` (channel, gate and body share of each gate's leakage) and
`results/partA/REGION_DIAGNOSTIC.md` (transistor-level error in the region the gates use).
