"""Part A data: four 10,000-row datasets (+ optional training-only blocks)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.devices import device_netlist, generate_case  # noqa: E402
from leakml.sampling import draw_device_inputs  # noqa: E402
EDGE_FRACTION = 0.8      # share of Vd / Vs draws taken from the rail boundary layer in edge blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=0, help="training-only blocks of 10,000 rows per case")
    ap.add_argument("--edge", type=int, default=0, help="training-only blocks with Vd/Vs concentrated at the rails")
    args = ap.parse_args()

    out = spec.ROOT / "data" / "partA"
    ext = spec.ROOT / "data" / "partA_extra"
    edge = spec.ROOT / "data" / "partA_edge"
    net = spec.ROOT / "netlists" / "partA"
    for d in (out, ext, edge, net):
        d.mkdir(parents=True, exist_ok=True)

    for case, (kind, _, _) in spec.CASES.items():
        f = out / f"{case}.csv"
        if not f.exists():
            df, failed = generate_case(case, 10000, spec.SEEDS["data"])
            df.to_csv(f, index=False)
            print(f"{case}: 10,000 rows, {failed} failed simulations redrawn")
        sample = draw_device_inputs(case, 1, spec.SEEDS["data"]).iloc[0].to_dict()
        (net / f"{case}.cir").write_text(device_netlist(kind, sample))
        for k in range(1, args.extra + 1):
            g = ext / f"{case}_{k}.csv"
            if not g.exists():
                df, failed = generate_case(case, 10000, spec.SEEDS["extra"] + k)
                df.to_csv(g, index=False)
                print(f"{case} extra block {k}: {failed} redrawn")
        for k in range(1, args.edge + 1):
            g = edge / f"{case}_{k}.csv"
            if not g.exists():
                df, failed = generate_case(case, 10000, spec.SEEDS["extra"] + 1000 + k, edge_fraction=EDGE_FRACTION)
                df.to_csv(g, index=False)
                print(f"{case} edge block {k}: {failed} redrawn")


if __name__ == "__main__":
    main()
