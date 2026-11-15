"""8-bit parity checker: a balanced tree of seven XOR2 stages.

PARITY8 builds each XOR2 from three library gates, XOR(a, b) = AND2(OR2(a, b), NAND2(a, b)),
i.e. 6 + 4 + 6 = 16 transistors, 21 gate instances and 112 transistors in total.
PARITY8_NAND builds each XOR2 from four NAND2 (n = NAND(a, b), XOR = NAND(NAND(a, n), NAND(b, n))),
also 16 transistors per XOR2: 28 gate instances and 112 transistors.
The output p is 1 when an odd number of the inputs i0..i7 is 1.
"""
from typing import Sequence

from .circuits import Composite, Instance


def xor2(prefix: str, a: str, b: str, out: str) -> tuple:
    return (Instance(f"{prefix}o", "OR2", (a, b), f"{prefix}_or"),
            Instance(f"{prefix}n", "NAND2", (a, b), f"{prefix}_nd"),
            Instance(f"{prefix}a", "AND2", (f"{prefix}_or", f"{prefix}_nd"), out))


def xor2_nand(prefix: str, a: str, b: str, out: str) -> tuple:
    return (Instance(f"{prefix}m", "NAND2", (a, b), f"{prefix}_m"),
            Instance(f"{prefix}p", "NAND2", (a, f"{prefix}_m"), f"{prefix}_p"),
            Instance(f"{prefix}q", "NAND2", (b, f"{prefix}_m"), f"{prefix}_q"),
            Instance(f"{prefix}r", "NAND2", (f"{prefix}_p", f"{prefix}_q"), out))


def _tree(xor=xor2, name="parity8") -> Composite:
    inputs = tuple(f"i{k}" for k in range(8))
    nets, instances, level = list(inputs), [], 0
    while len(nets) > 1:
        nxt = []
        for k, (a, b) in enumerate(zip(nets[0::2], nets[1::2])):
            out = "p" if len(nets) == 2 else f"l{level}_{k}"
            instances += xor(f"x{level}{k}", a, b, out)
            nxt.append(out)
        nets, level = nxt, level + 1
    return Composite(name, inputs, tuple(instances), "p")


PARITY8 = _tree()
PARITY8_NAND = _tree(xor2_nand, "parity8_nand")


def parity_expected(bits: Sequence[int]) -> int:
    return sum(bits) % 2


def vector_bits(v: int) -> tuple:
    """Input vector number 0..255 -> bits (i0, ..., i7), i0 being the least significant."""
    return tuple((v >> k) & 1 for k in range(8))
