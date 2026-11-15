# Part C: 8-bit parity checker, ML vs SPICE

51,200 runs (200 corners x 256 input vectors). ML leakage = sum of the 21 gate-model leakages (OR2, NAND2, AND2 per XOR2; gate inputs at their logic values, loading ignored; internal nodes solved by the transistor models, no circuit-level SPICE data).

| Ones in input | runs | SPICE median (A) | ML median (A) | bias | median err | p95 err | max err | within 1 % | mean err |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | 4.651e-07 | 4.658e-07 | +0.02 % | 0.36 % | 2.14 % | 5.7 % | 82.0 % | 0.63 % |
| 1 | 1600 | 4.533e-07 | 4.532e-07 | +0.02 % | 0.38 % | 2.36 % | 5.5 % | 80.5 % | 0.65 % |
| 2 | 5600 | 4.376e-07 | 4.409e-07 | +0.06 % | 0.40 % | 2.41 % | 5.6 % | 79.6 % | 0.68 % |
| 3 | 11200 | 4.422e-07 | 4.425e-07 | +0.07 % | 0.40 % | 2.49 % | 5.4 % | 78.6 % | 0.69 % |
| 4 | 14000 | 4.267e-07 | 4.296e-07 | +0.10 % | 0.42 % | 2.54 % | 5.4 % | 77.7 % | 0.71 % |
| 5 | 11200 | 4.321e-07 | 4.311e-07 | +0.09 % | 0.43 % | 2.57 % | 5.2 % | 77.5 % | 0.72 % |
| 6 | 5600 | 4.180e-07 | 4.190e-07 | +0.12 % | 0.45 % | 2.60 % | 5.3 % | 76.9 % | 0.75 % |
| 7 | 1600 | 4.171e-07 | 4.182e-07 | +0.13 % | 0.45 % | 2.62 % | 5.0 % | 76.4 % | 0.75 % |
| 8 | 200 | 4.044e-07 | 4.036e-07 | +0.16 % | 0.47 % | 2.58 % | 5.1 % | 75.5 % | 0.76 % |
| all | 51200 | 4.318e-07 | 4.316e-07 | +0.09 % | 0.42 % | 2.52 % | 5.7 % | 78.0 % | 0.71 % |

**Overall mean accuracy: 99.29 %** (mean error 0.71 %).
