# What the gate leakage consists of (32nm HP)

Exact split of each gate case's leakage (SPICE device currents, first 40 reported corners): channel = drain/source current of the OFF devices at the cut (sub-threshold), gate = gate tunnelling (including the current drawn by inputs held at VDD), body = junction/body current. Shares are medians of component / total.

| Gate | Input | median total (A) | channel | gate | body |
|---|---|---|---|---|---|
| INV | 0 | 1.217e-08 | 99.8 % | 0.2 % | 0.00 % |
| INV | 1 | 9.725e-09 | 98.8 % | 1.2 % | 0.00 % |
| NAND2 | 00 | 2.804e-10 | 82.2 % | 17.8 % | 0.00 % |
| NAND2 | 01 | 1.226e-08 | 98.3 % | 1.7 % | 0.00 % |
| NAND2 | 10 | 3.071e-09 | 99.1 % | 0.9 % | 0.00 % |
| NAND2 | 11 | 1.943e-08 | 98.8 % | 1.2 % | 0.00 % |
| NOR2 | 00 | 2.434e-08 | 99.8 % | 0.2 % | 0.00 % |
| NOR2 | 01 | 9.711e-09 | 98.3 % | 1.7 % | 0.00 % |
| NOR2 | 10 | 1.788e-09 | 91.0 % | 9.0 % | 0.00 % |
| NOR2 | 11 | 3.158e-10 | 9.7 % | 90.2 % | 0.01 % |
| AND2 | 00 | 1.091e-08 | 98.1 % | 1.9 % | 0.00 % |
| AND2 | 01 | 2.516e-08 | 98.5 % | 1.5 % | 0.00 % |
| AND2 | 10 | 1.568e-08 | 98.8 % | 1.2 % | 0.00 % |
| AND2 | 11 | 3.763e-08 | 99.0 % | 1.0 % | 0.00 % |
| OR2 | 00 | 4.005e-08 | 99.3 % | 0.7 % | 0.00 % |
| OR2 | 01 | 2.508e-08 | 99.1 % | 0.9 % | 0.00 % |
| OR2 | 10 | 1.449e-08 | 98.4 % | 1.6 % | 0.00 % |
| OR2 | 11 | 1.243e-08 | 96.9 % | 3.1 % | 0.00 % |
