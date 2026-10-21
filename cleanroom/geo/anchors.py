"""Gameplay anchors: the only level facts kept from retail (spawn, warp and
trigger coordinates, path nodes, collision surface types the code
hard-wires). New geometry is built around them.

    Anchor(kind, pos, radius=0, meta={})
    extract_c(text, patterns) -> [Anchor]    regex extractor for decomp C/asm macros
    extract(paths, patterns)  -> [Anchor]
    register(name, fn) / EXTRACTORS          game-specific extractors plug in here
    save(path, anchors) / load(path) / points(anchors) -> (N, 3) array

A pattern is {"kind": "spawn", "regex": "...", "xyz": [1, 2, 3], "radius": 0};
the groups named in xyz are parsed as numbers (int, hex or float).
"""
import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np

EXTRACTORS = {}


@dataclass
class Anchor:
    kind: str
    pos: tuple
    radius: float = 0.0
    meta: dict = field(default_factory=dict)


def register(name, fn):
    """fn(paths) -> [Anchor]; a game's extractor (e.g. one that reads the
    decomp's level scripts) registers itself under the game id."""
    EXTRACTORS[name] = fn


def _num(s):
    s = s.strip().rstrip("fF")
    return float(int(s, 0)) if re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", s) else float(s)


def extract_c(text, patterns):
    out = []
    for p in patterns:
        for m in re.finditer(p["regex"], text):
            xyz = tuple(_num(m.group(g)) for g in p["xyz"])
            meta = {k: m.group(g) for k, g in p.get("meta", {}).items()}
            out.append(Anchor(p["kind"], xyz, p.get("radius", 0.0), meta))
    return out


def extract(paths, patterns):
    out = []
    for path in paths:
        with open(path, encoding="utf8", errors="replace") as f:
            for a in extract_c(f.read(), patterns):
                a.meta.setdefault("file", str(path))
                out.append(a)
    return out


def save(path, anchors):
    json.dump({"anchors": [asdict(a) for a in anchors]}, open(path, "w"), indent=1)


def load(path):
    d = json.load(open(path))
    return [Anchor(a["kind"], tuple(a["pos"]), a.get("radius", 0.0), a.get("meta", {})) for a in d["anchors"]]


def points(anchors):
    return np.array([a.pos for a in anchors], np.float64).reshape(-1, 3)
