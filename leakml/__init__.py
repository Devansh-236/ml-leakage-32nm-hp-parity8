import os

# torch and LightGBM each bundle an OpenMP runtime; with several OpenMP threads the two
# conflict on macOS (segfaults when unpickling a LightGBM model after torch was imported,
# or when torch trains after LightGBM). One OpenMP thread avoids it. Set before either import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
