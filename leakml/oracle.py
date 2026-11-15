"""SPICE-backed stand-in for the four transistor models. Same predict(df) -> (n, 4)
contract as CaseModel, but every row is simulated exactly. Used only in tests, to check
the circuit-level composition independently of any ML error."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import spec, spice
from .devices import device_netlist, parse_device


class _Oracle:
    def __init__(self, case):
        self.case, self.kind = case, spec.CASES[case][0]

    def _one(self, row):
        res = parse_device(spice.run_netlist(device_netlist(self.kind, row)).stdout)
        if res is None:
            raise RuntimeError(f"oracle SPICE run failed for {self.case}: {row}")
        return [res[t] for t in spec.CURRENTS]

    def predict(self, df) -> np.ndarray:
        rows = df.to_dict("records")
        with ThreadPoolExecutor(max_workers=10) as pool:
            return np.array(list(pool.map(self._one, rows)))


def OracleModels() -> dict:
    return {case: _Oracle(case) for case in spec.CASES}
