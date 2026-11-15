import pytest

from leakml.cards import parse_pm


def test_nominal_values():
    c = parse_pm()
    assert c["nmos"].params["vth0"] == 0.49396
    assert c["pmos"].params["vth0"] == -0.49155
    assert c["nmos"].params["toxe"] == 1.15e-9
    assert c["pmos"].params["ndep"] == 3.07e18


def test_tab_separated_xl_parsed():
    c = parse_pm()
    assert c["nmos"].params["xl"] == -14e-9
    assert c["pmos"].params["xl"] == -14e-9


def test_override_and_unknown():
    n = parse_pm()["nmos"]
    m = n.override({"vth0": 0.5})
    assert m.params["vth0"] == 0.5
    assert n.params["vth0"] == 0.49396
    assert {k: v for k, v in m.params.items() if k != "vth0"} == {k: v for k, v in n.params.items() if k != "vth0"}
    with pytest.raises(KeyError):
        n.override({"nope": 1.0})


def test_render_roundtrip(tmp_path):
    c = parse_pm()
    f = tmp_path / "x.pm"
    f.write_text(c["nmos"].render() + c["pmos"].render())
    d = parse_pm(f)
    assert d["nmos"].params == c["nmos"].params
    assert d["pmos"].params == c["pmos"].params
