# Fresh-corner check

200 corners drawn with seed 3207, used nowhere else in the project. New SPICE references for every gate case and for the parity checker (all 256 vectors); the final models are unchanged.

| Set | runs | median | p95 | max | within 1 % | mean error | mean accuracy |
|---|---|---|---|---|---|---|---|
| gates (a) | 3,600 | 0.43 % | 2.73 % | 23.8 % | 76.2 % | 0.79 % | 99.21 % |
| gates (b) | 3,600 | 0.52 % | 3.18 % | 20.9 % | 71.6 % | 0.92 % | 99.08 % |
| parity checker | 51,200 | 0.43 % | 2.09 % | 4.9 % | 79.1 % | 0.65 % | 99.35 % |
