"""Example netlists: one per gate (Part B) and the parity checker (Part C), at one random corner."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.circuits import GATES, circuit_netlist, flatten  # noqa: E402
from leakml.parity import PARITY8  # noqa: E402
from leakml.sampling import draw_corners  # noqa: E402


def main():
    corner = draw_corners(1, spec.SEEDS["corners_report"]).iloc[0].to_dict()
    b, c = spec.ROOT / "netlists" / "partB", spec.ROOT / "netlists" / "partC"
    b.mkdir(parents=True, exist_ok=True)
    c.mkdir(parents=True, exist_ok=True)
    for name, gate in GATES.items():
        (b / f"{name}.cir").write_text(circuit_netlist(gate, corner, (1,) * len(gate.inputs)))
    (c / "parity8.cir").write_text(circuit_netlist(flatten(PARITY8), corner, (1, 0, 1, 1, 0, 0, 1, 0)))
    print("wrote", len(GATES), "gate netlists and parity8.cir")


if __name__ == "__main__":
    main()
