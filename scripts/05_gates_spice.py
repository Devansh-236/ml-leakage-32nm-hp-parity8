"""Part B SPICE reference: 18 gate cases x 200 corners, for the reported and the development corners."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakml import spec  # noqa: E402
from leakml.reference import gate_reference  # noqa: E402
from leakml.sampling import draw_corners  # noqa: E402


def main():
    out = spec.ROOT / "data" / "partB"
    out.mkdir(parents=True, exist_ok=True)
    for tag, seed in (("report", spec.SEEDS["corners_report"]), ("dev", spec.SEEDS["corners_dev"])):
        corners = draw_corners(spec.N_CORNERS, seed)
        corners.to_csv(out / f"corners_{tag}.csv", index=False)
        df = gate_reference(corners)
        df.to_csv(out / f"gates_spice_{tag}.csv", index=False)
        print(f"{tag}: {len(df)} runs, median leakage {df.leak.median():.3e} A")


if __name__ == "__main__":
    main()
