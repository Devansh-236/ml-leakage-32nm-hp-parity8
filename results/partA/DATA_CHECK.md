# Data check (Part A)

## Kirchhoff residual

Residual = `|id+ig+is+ib|` per row (10,000 rows per case). A row passes when `residual <= 1e-6 * max(|id|,|ig|,|is|,|ib|) + 2e-15 A`: the relative term covers normal rows, the 2e-15 A term is the simulator's absolute noise floor (BSIM4 body currents sit at about 1e-14 A), which matters only where every terminal current is tiny, e.g. Vds close to 0.

| Case | rows | max abs residual (A) | median abs residual (A) | max rel. residual | rows failing the rule |
|---|---|---|---|---|---|
| NMOS_ON | 10000 | 1.35e-15 | 1.11e-16 | 2.94e-02 | 0 |
| NMOS_OFF | 10000 | 5.38e-20 | 2.56e-23 | 4.94e-12 | 0 |
| PMOS_ON | 10000 | 6.40e-16 | 3.55e-17 | 7.17e-03 | 0 |
| PMOS_OFF | 10000 | 1.42e-15 | 9.26e-17 | 1.28e-01 | 0 |

## Trends (one parameter swept, others nominal, VDD = 1 V, W = 32 nm)

- `trend_off_vs_temp.png`, `trend_off_vs_vth0.png`, `trend_gate_vs_toxe.png`
- NMOS_OFF: |id| 4.19e-10 A at −40 °C, 8.30e-09 A at 125 °C (×20).
- PMOS_OFF: |id| 1.98e-10 A at −40 °C, 4.22e-09 A at 125 °C (×21).
- NMOS_OFF: |id| changes ×0.08 from vth0 −10 % to +10 %.
- PMOS_OFF: |id| changes ×0.058 from vth0 −10 % to +10 %.
- NMOS_ON: |ig| changes ×0.057 from toxe −10 % to +10 %.
- PMOS_ON: |ig| changes ×0.088 from toxe −10 % to +10 %.
