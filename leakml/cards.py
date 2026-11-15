"""Parse the professor's PTM/BSIM4 model cards and render modified copies."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .spec import MODEL_FILE

_HEADER = re.compile(r"^\.model\s+(\S+)\s+(\S+)\s+level\s*=\s*(\S+)", re.I)
_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


@dataclass
class Card:
    name: str
    kind: str                       # "nmos" or "pmos"
    level: str
    params: dict = field(default_factory=dict)

    def override(self, values: Mapping[str, float], name=None) -> "Card":
        unknown = sorted(set(values) - set(self.params))
        if unknown:
            raise KeyError(f"parameters not in the {self.name} card: {unknown}")
        merged = {**self.params, **{k: float(v) for k, v in values.items()}}
        return Card(name or self.name, self.kind, self.level, merged)

    def render(self) -> str:
        pairs = [f"{k} = {float(v)!r}" for k, v in self.params.items()]
        rows = ["+ " + "   ".join(pairs[i:i + 4]) for i in range(0, len(pairs), 4)]
        return "\n".join([f".model {self.name} {self.kind} level = {self.level}", *rows]) + "\n"


def parse_pm(path: Path = MODEL_FILE) -> dict:
    """Return {"nmos": Card, "pmos": Card} (keyed by device kind)."""
    cards, cur = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        m = _HEADER.match(line)
        if m:
            name, kind, level = m.groups()
            cur = Card(name, kind.lower(), level)
            cards[cur.kind] = cur
        elif line.startswith("+") and cur is not None:
            for k, v in _PAIR.findall(line):
                cur.params[k.lower()] = float(v)
    return cards
