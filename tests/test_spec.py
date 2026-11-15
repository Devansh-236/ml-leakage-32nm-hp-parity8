from leakml import spec


def test_assignment_constants():
    assert spec.VDD_RANGE == (0.9, 1.1)
    assert spec.L == 32e-9
    assert spec.W_RANGE == (32e-9, 256e-9)
    assert spec.CASES["PMOS_ON"] == ("pmos", 0, 1)
    assert spec.CASES["NMOS_OFF"] == ("nmos", 0, 0)
    assert spec.OPTIONS == ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
    assert sum(spec.SPLIT) == 10000
    assert len(set(spec.SEEDS.values())) == len(spec.SEEDS)
