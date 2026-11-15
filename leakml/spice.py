"""ngspice discovery, batch runs and output parsing."""
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

_PRINT = re.compile(r"^([iv]\([^)]+\))\s*=\s*([-+0-9.eE]+)\s*$", re.M | re.I)


def find_ngspice() -> str:
    path = os.environ.get("NGSPICE") or shutil.which("ngspice")
    if not path:
        raise RuntimeError("ngspice not found: install it (e.g. `brew install ngspice`) "
                           "or point the NGSPICE environment variable at the binary")
    return path


def ngspice_version() -> str:
    out = subprocess.run([find_ngspice(), "-v"], capture_output=True, text=True).stdout
    m = re.search(r"ngspice-(\d+(?:\.\d+)*)", out)
    return m.group(1) if m else out.strip().splitlines()[0]


@dataclass
class SpiceResult:
    stdout: str
    ok: bool
    error: Optional[str]


def run_netlist(text: str, timeout: float = 120) -> SpiceResult:
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        proc = subprocess.run([find_ngspice(), "-b", path], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return SpiceResult("", False, "timeout")
    finally:
        os.unlink(path)
    ok = proc.returncode == 0
    return SpiceResult(proc.stdout, ok, None if ok else proc.stderr.strip()[-300:])


def _one(text):
    return run_netlist(text)


def run_many(texts: Sequence[str], workers: Optional[int] = None) -> list:
    """Run netlists in parallel; results come back in input order."""
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return list(pool.map(_one, texts, chunksize=16))


def parse_prints(stdout: str) -> dict:
    """{"i(vd)": value, "v(x)": value, ...} with lower-case keys."""
    return {k.lower(): float(v) for k, v in _PRINT.findall(stdout)}
