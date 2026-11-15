"""Check the simulator and the nominal operating points before generating anything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spice  # noqa: E402
from leakml.cards import parse_pm  # noqa: E402
from leakml.devices import run_device  # noqa: E402
from leakml.spec import PROCESS_PARAMS  # noqa: E402

print("ngspice", spice.ngspice_version(), "at", spice.find_ngspice())
cards = parse_pm()
nom = {k: {p: c.params[p] for p in PROCESS_PARAMS} for k, c in cards.items()}
pts = {
    "NMOS_ON": ("nmos", dict(vd=1.0, vg=1.0, vs=0.0, vb=0.0)),
    "NMOS_OFF": ("nmos", dict(vd=1.0, vg=0.0, vs=0.0, vb=0.0)),
    "PMOS_ON": ("pmos", dict(vd=0.0, vg=0.0, vs=1.0, vb=1.0)),
    "PMOS_OFF": ("pmos", dict(vd=0.0, vg=1.0, vs=1.0, vb=1.0)),
}
ok = True
for name, (kind, v) in pts.items():
    cur = run_device(kind, {**v, "temp": 27.0, "w": 32e-9, **nom[kind]})
    resid = abs(sum(cur.values())) / max(abs(x) for x in cur.values())
    ok &= resid < 1e-6
    print(f"{name:9s}", {k: f"{x:.3e}" for k, x in cur.items()}, f"KCL residual {resid:.1e}")
sys.exit(0 if ok else 1)
