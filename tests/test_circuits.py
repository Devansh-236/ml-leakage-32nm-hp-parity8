import itertools

import pytest

from leakml import spec
from leakml.circuits import GATES, circuit_netlist, logic, simulate_circuit
from leakml.sampling import nominal_corner


def test_transistor_counts():
    assert {g: len(c.fets) for g, c in GATES.items()} == {"INV": 2, "NAND2": 4, "NOR2": 4, "AND2": 6, "OR2": 6}


def test_internal_nodes():
    assert GATES["INV"].internal == ()
    assert GATES["NAND2"].internal == ("x",)
    assert GATES["NOR2"].internal == ("y",)
    assert GATES["AND2"].internal == ("x", "n")
    assert GATES["OR2"].internal == ("y", "n")


@pytest.mark.parametrize("gate,fn", [("INV", lambda a, b: 1 - a), ("NAND2", lambda a, b: 1 - (a & b)),
                                      ("NOR2", lambda a, b: 1 - (a | b)), ("AND2", lambda a, b: a & b),
                                      ("OR2", lambda a, b: a | b)])
def test_logic_truth_tables(gate, fn):
    n = len(GATES[gate].inputs)
    for bits in itertools.product((0, 1), repeat=n):
        assert logic(gate, bits) == fn(bits[0], bits[-1])


def test_netlist_has_required_options():
    text = circuit_netlist(GATES["NAND2"], nominal_corner(), (1, 0))
    assert spec.OPTIONS in text and ".options temp=" in text
    assert "Vdd vdd 0" in text and "Va a 0" in text


def test_nand2_11_state():
    r = simulate_circuit(GATES["NAND2"], nominal_corner(), (1, 1))
    assert r["out"] < 0.05 and r["x"] < 0.05 and r["leak"] > 0


def test_nor2_00_state():
    r = simulate_circuit(GATES["NOR2"], nominal_corner(), (0, 0))
    assert r["out"] > 0.95 and 0.9 < r["y"] < 1.01 and r["leak"] > 0


def test_input_pin_included():
    r = simulate_circuit(GATES["AND2"], nominal_corner(), (1, 1))
    assert r["pin_a"] != 0 and r["pin_b"] != 0
    assert r["leak"] == pytest.approx(r["iddd"] + r["pin_a"] + r["pin_b"], rel=1e-12)
    z = simulate_circuit(GATES["AND2"], nominal_corner(), (0, 0))
    assert z["leak"] == pytest.approx(z["iddd"], rel=1e-12)   # inputs at 0 V add nothing


def test_every_gate_every_pattern_runs():
    for g, c in GATES.items():
        for bits in itertools.product((0, 1), repeat=len(c.inputs)):
            r = simulate_circuit(c, nominal_corner(), bits)
            assert r is not None and r["leak"] > 0, (g, bits)
            assert r["out"] == pytest.approx(logic(g, bits) * 1.0, abs=0.05)
