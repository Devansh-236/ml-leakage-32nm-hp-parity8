import pytest

from leakml import spice
from leakml.cards import parse_pm
from leakml.devices import device_netlist, run_device

NOM = {k: parse_pm()["nmos"].params[k] for k in ("vth0", "toxe", "ndep", "u0", "xj")}
PNOM = {k: parse_pm()["pmos"].params[k] for k in ("vth0", "toxe", "ndep", "u0", "xj")}


def point(case):
    kind = case.split("_")[0].lower()
    on = case.endswith("ON")
    if kind == "nmos":
        return kind, dict(vd=1.0, vg=1.0 if on else 0.0, vs=0.0, vb=0.0, temp=27.0, w=32e-9, **NOM)
    return kind, dict(vd=0.0, vg=0.0 if on else 1.0, vs=1.0, vb=1.0, temp=27.0, w=32e-9, **PNOM)


def test_find_ngspice_honours_env(monkeypatch):
    monkeypatch.setenv("NGSPICE", "/some/where/ngspice")
    assert spice.find_ngspice() == "/some/where/ngspice"
    monkeypatch.delenv("NGSPICE")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="ngspice"):
        spice.find_ngspice()


@pytest.mark.parametrize("case,term,expected", [
    ("NMOS_ON", "id", -4.30e-5), ("NMOS_ON", "ig", -5.44e-11), ("NMOS_OFF", "id", -1.896e-9),
    ("PMOS_ON", "id", 3.003e-5), ("PMOS_OFF", "id", 9.927e-10)])
def test_nominal_points(case, term, expected):
    kind, p = point(case)
    got = run_device(kind, p)
    assert got[term] == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("case", ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"])
def test_kirchhoff(case):
    kind, p = point(case)
    got = run_device(kind, p)
    total = sum(got.values())
    assert abs(total) < 1e-6 * max(abs(v) for v in got.values())


def test_netlist_requirements():
    kind, p = point("NMOS_OFF")
    text = device_netlist(kind, p)
    assert ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9" in text
    for src in ("Vd ", "Vg ", "Vs ", "Vb "):
        assert src in text


def test_run_many_preserves_order():
    kind, p = point("NMOS_OFF")
    texts = [device_netlist(kind, {**p, "vd": 0.1 * (i + 1)}) for i in range(8)]
    res = spice.run_many(texts, workers=4)
    ids = [abs(spice.parse_prints(r.stdout)["i(vd)"]) for r in res]
    assert ids == sorted(ids)


def test_failure_returns_none(monkeypatch):
    kind, p = point("NMOS_OFF")
    assert run_device(kind, {**p, "w": float("nan")}) is None
