import numpy as np
import pandas as pd
import pytest

from leakml.analysis import bin_errors, leakage_components


def test_bin_errors_known_values():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3, 1.5, 1.6, 9.0], "err": [0.01, 0.03, 0.02, 0.10, 0.20, 0.50]})
    t = bin_errors(df, "x", [0, 1, 2, 3])
    assert t["n"].tolist() == [3, 2, 0]
    assert t["median"].iloc[0] == pytest.approx(0.02) and t["median"].iloc[1] == pytest.approx(0.15)
    assert np.isnan(t["median"].iloc[2]) and np.isnan(t["p95"].iloc[2])      # empty bin: NaN, no exception
    assert len(t) == 3 and (t.index.astype(str)[0].startswith("["))


def test_leakage_components_sum_to_total_for_nmos_off():
    # an OFF NMOS at the supply rails: channel current dominates, components are non-negative shares
    from leakml.sampling import nominal_corner
    c = leakage_components("INV", (1,), nominal_corner())
    assert set(c) == {"channel", "gate", "body", "total"}
    assert c["total"] == pytest.approx(c["channel"] + c["gate"] + c["body"], rel=1e-9)
    assert c["channel"] > 0 and all(np.isfinite(v) for v in c.values())


def test_components_total_matches_spice_leakage():
    from leakml.circuits import GATES, simulate_circuit
    from leakml.sampling import draw_corners
    corner = draw_corners(1, 5).iloc[0].to_dict()
    for gate, bits in (("NAND2", (1, 0)), ("OR2", (0, 0)), ("INV", (0,))):
        c = leakage_components(gate, bits, corner)
        assert c["total"] == pytest.approx(simulate_circuit(GATES[gate], corner, bits)["leak"], rel=1e-3)


def test_region_errors_keeps_only_the_gate_relevant_region():
    from leakml.analysis import region_errors

    class Fake:                                  # predicts exactly 10 % too high on id
        def predict(self, df):
            out = np.zeros((len(df), 4))
            out[:, 0] = 1.1 * df["id"].to_numpy()
            return out

    df = pd.DataFrame({"vdd": [1.0, 1.0, 1.0], "vs": [0.05, 0.5, 0.1], "vd": [0.9, 0.9, 0.2], "id": [2e-9, 3e-9, 4e-9]})
    out = region_errors("NMOS_OFF", Fake(), df)  # region: vs < 0.2 VDD and vd > 0.7 VDD -> only row 0
    assert len(out) == 1 and out["err"].iloc[0] == pytest.approx(0.1)
    both = region_errors("PMOS_OFF", Fake(), pd.DataFrame({"vdd": [1.0], "vs": [0.9], "vd": [0.1], "id": [-1e-9]}))
    assert len(both) == 1 and both["err"].iloc[0] == pytest.approx(0.1)
