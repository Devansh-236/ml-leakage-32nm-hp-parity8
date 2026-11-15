"""Transistor-level circuit description, the five-gate library and netlist builder.

Nodes are lower-case strings. "vdd" is the supply, "0" ground; a gate's output node
is "out". Every netlist (single gate or whole circuit) is built here, so gate-level
and circuit-level SPICE cannot disagree about the transistors.

Leakage = current delivered by the VDD source + current delivered by every input
source held at VDD (leakage power / VDD; an input at 0 V delivers no power).
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from . import spice
from .cards import parse_pm
from .devices import num
from .spec import L, OPTIONS, PROCESS_PARAMS


@dataclass(frozen=True)
class Fet:
    name: str
    kind: str       # "nmos" / "pmos"
    d: str
    g: str
    s: str
    b: str


@dataclass(frozen=True)
class Circuit:
    name: str
    inputs: tuple
    fets: tuple
    internal: tuple     # internal nodes whose voltages are recorded (excluding the output)
    output: str


def _inv(pre, a, out):
    return [Fet(f"{pre}p", "pmos", out, a, "vdd", "vdd"), Fet(f"{pre}n", "nmos", out, a, "0", "0")]


def _nand(pre, a, b, out, x):
    return [Fet(f"{pre}pa", "pmos", out, a, "vdd", "vdd"), Fet(f"{pre}pb", "pmos", out, b, "vdd", "vdd"),
            Fet(f"{pre}na", "nmos", out, a, x, "0"),      # top of the series stack
            Fet(f"{pre}nb", "nmos", x, b, "0", "0")]


def _nor(pre, a, b, out, y):
    return [Fet(f"{pre}pa", "pmos", y, a, "vdd", "vdd"),    # top of the series stack
            Fet(f"{pre}pb", "pmos", out, b, y, "vdd"),
            Fet(f"{pre}na", "nmos", out, a, "0", "0"), Fet(f"{pre}nb", "nmos", out, b, "0", "0")]


GATES = {
    "INV": Circuit("INV", ("a",), tuple(_inv("", "a", "out")), (), "out"),
    "NAND2": Circuit("NAND2", ("a", "b"), tuple(_nand("", "a", "b", "out", "x")), ("x",), "out"),
    "NOR2": Circuit("NOR2", ("a", "b"), tuple(_nor("", "a", "b", "out", "y")), ("y",), "out"),
    "AND2": Circuit("AND2", ("a", "b"), tuple(_nand("g_", "a", "b", "n", "x") + _inv("i_", "n", "out")),
                    ("x", "n"), "out"),
    "OR2": Circuit("OR2", ("a", "b"), tuple(_nor("g_", "a", "b", "n", "y") + _inv("i_", "n", "out")),
                   ("y", "n"), "out"),
}


def logic(gate: str, bits: Sequence[int]) -> int:
    a, b = bits[0], bits[-1]
    return {"INV": 1 - a, "NAND2": 1 - (a & b), "NOR2": 1 - (a | b), "AND2": a & b, "OR2": a | b}[gate]


def model_text(corner: Mapping) -> str:
    cards = parse_pm()
    n = cards["nmos"].override({p: corner[f"n_{p}"] for p in PROCESS_PARAMS}, name="nch")
    p = cards["pmos"].override({p: corner[f"p_{p}"] for p in PROCESS_PARAMS}, name="pch")
    return n.render() + p.render()


def fet_line(f: Fet, corner: Mapping) -> str:
    model, w = ("nch", corner["wn"]) if f.kind == "nmos" else ("pch", corner["wp"])
    return f"M{f.name} {f.d} {f.g} {f.s} {f.b} {model} W={num(w)} L={num(L)}"


def probe_nodes(c: Circuit) -> tuple:
    return tuple(c.internal) + (c.output,)


def circuit_netlist(c: Circuit, corner: Mapping, bits: Sequence[int]) -> str:
    vdd = corner["vdd"]
    lines = [f"* {c.name} leakage, inputs {''.join(map(str, bits))}", OPTIONS,
             f".options temp={num(corner['temp'])}", model_text(corner),
             *[fet_line(f, corner) for f in c.fets], f"Vdd vdd 0 {num(vdd)}"]
    lines += [f"V{n} {n} 0 {num(vdd * v)}" for n, v in zip(c.inputs, bits)]
    probes = " ".join([f"i(v{n})" for n in ("dd",) + tuple(c.inputs)] + [f"v({n})" for n in probe_nodes(c)])
    lines += [".control", "set numdgt=12", "op", f"print {probes}", ".endc", ".end", ""]
    return "\n".join(lines)


def parse_circuit(c: Circuit, stdout: str, bits: Sequence[int]) -> Optional[dict]:
    vals = spice.parse_prints(stdout)
    need = ["i(vdd)"] + [f"i(v{n})" for n in c.inputs] + [f"v({n})" for n in probe_nodes(c)]
    if not all(k in vals for k in need):
        return None
    res = {"iddd": -vals["i(vdd)"]}
    for n in c.inputs:
        res[f"pin_{n}"] = -vals[f"i(v{n})"]
    res["leak"] = res["iddd"] + sum(res[f"pin_{n}"] * b for n, b in zip(c.inputs, bits))
    for n in probe_nodes(c):
        res[n] = vals[f"v({n})"]
    return res


def simulate_circuit(c: Circuit, corner: Mapping, bits: Sequence[int]) -> Optional[dict]:
    """Run one circuit at one corner and input vector; None if ngspice failed."""
    return parse_circuit(c, spice.run_netlist(circuit_netlist(c, corner, bits)).stdout, bits)


@dataclass(frozen=True)
class Instance:
    name: str
    gate: str
    inputs: tuple
    output: str


@dataclass(frozen=True)
class Composite:
    name: str
    inputs: tuple
    instances: tuple
    output: str


def flatten(comp: Composite) -> Circuit:
    """Transistor-level circuit of a composite. An instance's internal nodes become
    "<instance>_<node>", its FETs "<instance>_<fet>"."""
    fets, internal = [], []
    for inst in comp.instances:
        g = GATES[inst.gate]
        rename = {"vdd": "vdd", "0": "0", g.output: inst.output,
                  **dict(zip(g.inputs, inst.inputs)), **{n: f"{inst.name}_{n}" for n in g.internal}}
        for f in g.fets:
            fets.append(Fet(f"{inst.name}_{f.name}", f.kind, rename[f.d], rename[f.g], rename[f.s], rename[f.b]))
        internal += [rename[n] for n in g.internal]
        if inst.output != comp.output:
            internal.append(inst.output)
    return Circuit(comp.name, comp.inputs, tuple(fets), tuple(internal), comp.output)


def net_values(comp: Composite, bits: Sequence[int]) -> dict:
    """Logic value of every net for an input vector."""
    v = dict(zip(comp.inputs, bits))
    for inst in comp.instances:
        v[inst.output] = logic(inst.gate, [v[n] for n in inst.inputs])
    return v
