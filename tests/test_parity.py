import itertools
import random

import pytest

from leakml.circuits import flatten, net_values, simulate_circuit
from leakml.parity import PARITY8, parity_expected
from leakml.sampling import nominal_corner


def test_parity_truth_table():
    for bits in itertools.product((0, 1), repeat=8):
        assert net_values(PARITY8, bits)["p"] == sum(bits) % 2 == parity_expected(bits)


def test_structure():
    gates = [i.gate for i in PARITY8.instances]
    assert len(gates) == 21
    assert {g: gates.count(g) for g in set(gates)} == {"OR2": 7, "NAND2": 7, "AND2": 7}
    flat = flatten(PARITY8)
    assert len(flat.fets) == 112
    assert len({f.name for f in flat.fets}) == 112
    assert len(set(flat.internal)) == len(flat.internal)
    assert flat.inputs == tuple(f"i{k}" for k in range(8)) and flat.output == "p"


def test_spice_output_levels():
    flat = flatten(PARITY8)
    rng = random.Random(1)
    vectors = [(0,) * 8, (1,) * 8, (1, 0, 0, 0, 0, 0, 0, 0)] + [tuple(rng.randint(0, 1) for _ in range(8)) for _ in range(3)]
    for bits in vectors:
        r = simulate_circuit(flat, nominal_corner(), bits)
        assert r is not None
        assert abs(r["p"] - parity_expected(bits) * 1.0) < 0.1, bits


def test_nand_only_parity_structure_and_truth_table():
    from leakml.parity import PARITY8_NAND
    gates = [i.gate for i in PARITY8_NAND.instances]
    assert gates == ["NAND2"] * 28                         # 7 XOR2 x 4 NAND2
    flat = flatten(PARITY8_NAND)
    assert len(flat.fets) == 112 and len({f.name for f in flat.fets}) == 112
    assert flat.inputs == tuple(f"i{k}" for k in range(8)) and flat.output == "p"
    for v in range(256):
        bits = tuple((v >> k) & 1 for k in range(8))
        assert net_values(PARITY8_NAND, bits)["p"] == sum(bits) % 2
