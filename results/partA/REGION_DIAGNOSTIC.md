# Transistor-level error in the gate-relevant bias region (OFF cases)

8000 fresh SPICE rows per case (seed 3206, 80 % of Vd/Vs drawn next to the rails), never used for training or selection. Only rows inside the region gates use are kept (NMOS: Vs < 0.2 VDD, Vd > 0.7 VDD; PMOS mirrored). Error = |id_ML - id_SPICE| / |id_SPICE|; entries are median % (n).

## NMOS_OFF: 1604 rows in the region, median error 0.28 %, p95 1.2 %, median |id| 6.7e-09 A

- temperature (C): [-40, 0): 0.30 % (n=452); [0, 40): 0.20 % (n=354); [40, 80): 0.25 % (n=343); [80, 100): 0.31 % (n=197); [100, 125.1): 0.56 % (n=258)
- |vth0| quintiles (0.445 to 0.543 V): [0.445, 0.464): 0.33 % (n=321); [0.464, 0.484): 0.24 % (n=321); [0.484, 0.505): 0.23 % (n=320); [0.505, 0.526): 0.29 % (n=321); [0.526, 0.543): 0.32 % (n=321)
- width (nm): [32, 70): 0.30 % (n=281); [70, 110): 0.26 % (n=312); [110, 160): 0.29 % (n=363); [160, 210): 0.28 % (n=355); [210, 257): 0.30 % (n=293)
- VDD (V): [0.9, 0.95): 0.29 % (n=424); [0.95, 1): 0.26 % (n=396); [1, 1.05): 0.26 % (n=407); [1.05, 1.1): 0.32 % (n=377)
- split at the median |id|: above 0.29 %, below 0.27 %

## PMOS_OFF: 1629 rows in the region, median error 1.15 %, p95 4.4 %, median |id| 3.3e-09 A

- temperature (C): [-40, 0): 0.94 % (n=393); [0, 40): 1.00 % (n=409); [40, 80): 1.28 % (n=389); [80, 100): 1.42 % (n=202); [100, 125.1): 1.59 % (n=236)
- |vth0| quintiles (0.443 to 0.541 V): [0.443, 0.463): 1.31 % (n=326); [0.463, 0.481): 1.13 % (n=326); [0.481, 0.502): 1.02 % (n=325); [0.502, 0.523): 1.18 % (n=326); [0.523, 0.541): 1.20 % (n=326)
- width (nm): [32, 70): 1.57 % (n=287); [70, 110): 1.18 % (n=271); [110, 160): 1.08 % (n=384); [160, 210): 0.95 % (n=357); [210, 257): 1.22 % (n=330)
- VDD (V): [0.9, 0.95): 1.36 % (n=406); [0.95, 1): 1.06 % (n=430); [1, 1.05): 1.01 % (n=385); [1.05, 1.1): 1.23 % (n=408)
- split at the median |id|: above 1.16 %, below 1.15 %
