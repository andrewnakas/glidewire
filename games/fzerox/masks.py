"""The 38 u8[512] arrays in common_assets_compressed: 64x64 1-bit masks (ending fireworks shapes).

    masks.mask_bytes(symbol)     CLEAN: our own 64x64 firework shape per slot (seeded), packed MSB-first
(the older spec() reader of retail occupancy is kept only as dev history; spec/masks.json is no longer used)
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec", "masks.json")
RX = re.compile(r"u8 (D_F2[0-9A-F]{5})\[\] = \{([^}]*)\};")


def unpack(vals):
    b = np.array(vals, np.uint8)
    return np.unpackbits(b).reshape(64, 64)


def spec(tree):
    p = os.path.join(tree, "src", "assets", "us", "rev0", "common_assets_compressed", "common_assets_compressed.c")
    out = {}
    for m in RX.finditer(open(p).read()):
        vals = [int(v) for v in re.findall(r"\d+", m.group(2))]
        if len(vals) != 512:
            continue
        bits = unpack(vals).astype(np.float32)
        g = bits.reshape(16, 4, 16, 4).mean((1, 3))
        out[m.group(1)] = [int(round(v * 3)) for v in g.reshape(-1)]   # 2-bit coverage per 4x4 cell
    json.dump(out, open(SPEC, "w"), indent=0)
    print(f"masks: {len(out)} (16x16 2-bit coverage)")


def mask_bytes(sym):
    """Our 64x64 firework shape for this slot (2026-10 rebuild: no retail occupancy is read): a seeded burst,
    ring, star or spiral of dots, packed MSB-first like the decomp's u8[512] arrays."""
    from cleanroom.decomp.gen import h32
    rng = np.random.default_rng(h32("glidewire-mask", sym))
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    u, v = (xx - 31.5) / 32, (yy - 31.5) / 32
    r, a = np.sqrt(u * u + v * v), np.arctan2(v, u)
    kind = int(rng.integers(4))
    if kind == 0:
        k = int(rng.integers(6, 13))
        m = (r < 0.95) & (np.cos(a * k) > 0.6 - 0.8 * r) & (r > 0.15)
    elif kind == 1:
        m = (np.abs(r - rng.uniform(0.5, 0.8)) < rng.uniform(0.06, 0.14))
    elif kind == 2:
        k = int(rng.integers(4, 8))
        m = r < 0.35 + 0.6 * np.abs(np.cos(a * k / 2)) ** 4
    else:
        m = np.zeros_like(r, bool)
        for t in np.linspace(0, 1, int(rng.integers(10, 20))):
            ang, rad = t * 4 * np.pi + rng.uniform(0, 0.3), 0.1 + 0.8 * t
            m |= (u - rad * np.cos(ang)) ** 2 + (v - rad * np.sin(ang)) ** 2 < 0.004 + 0.006 * t
    bits = m.astype(np.uint8)
    return [int(v) for v in np.packbits(bits.reshape(-1))]
