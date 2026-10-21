"""DIRTY ROOM: Torch-extracted F-Zero X textures -> clean-room spec (coarse facts only).

    python -m games.fzerox.extract_spec <dirty tree> [--png D:/n64work/fzerox/dirty_png]

Textures: every TEXTURE / COMPRESSED_TEXTURE entry of assets/yaml/us/rev0/*.yaml. Torch writes the data to
src/assets/us/rev0/<file>/<symbol>.<fmt>.inc.c (plain) or .incbin.c (MIO0 bytes, decompressed here).
Facts kept: format, size, TLUT link, a colour grid (4x4, 16x16 for >= 128 px) and a 2-bit alpha outline
(cleanroom.decomp.spec). CI textures are decoded through their TLUT first (palettes are rebuilt by
generate.py). --png writes decoded dirty PNGs for contact sheets (dirty room only, never committed).
"""
import json
import os
import re
import sys

import crunch64
import numpy as np

from cleanroom.decomp.spec import texture_fact
from cleanroom.gfx import png, texfmt

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec")
VER = "us/rev0"

FMTS = {"RGBA16": (texfmt.RGBA, texfmt.B16), "RGBA32": (texfmt.RGBA, texfmt.B32),
        "IA8": (texfmt.IA, texfmt.B8), "IA16": (texfmt.IA, texfmt.B16), "IA4": (texfmt.IA, texfmt.B4),
        "I4": (texfmt.I, texfmt.B4), "I8": (texfmt.I, texfmt.B8),
        "CI4": (texfmt.CI, texfmt.B4), "CI8": (texfmt.CI, texfmt.B8), "TLUT": (texfmt.RGBA, texfmt.B16)}
CSIZE = {"u8": 1, "s8": 1, "u16": 2, "s16": 2, "u32": 4, "s32": 4, "u64": 8, "Gfx": 8}


def yaml_entries(path):
    """Block-style Torch entries ("name:\\n  type: TEXTURE\\n  width: ..."); commented lines skipped."""
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"^(\w+):[ \t]*\n((?:[ \t]+\w+:[^\n]*\n?)+)", txt, re.M):
        d = dict((k.strip(), v.strip()) for k, v in (ln.split(":", 1) for ln in m.group(2).splitlines() if ":" in ln))
        if "type" in d:
            out.append((m.group(1), d))
    return out


def inc_bytes(path, csize):
    vals = re.findall(r"0x[0-9A-Fa-f]+|\b\d+\b", open(path).read())
    return b"".join(int(v, 0).to_bytes(csize, "big") for v in vals)


def load_entries(tree):
    """-> {symbol: entry} with raw texel bytes in e['_data'] (decompressed for MIO0 textures)."""
    ydir = os.path.join(tree, "assets", "yaml", VER)
    tex, missing = {}, []
    for yf in sorted(os.listdir(ydir)):
        base = os.path.splitext(yf)[0]
        ents = [(n, d) for n, d in yaml_entries(os.path.join(ydir, yf))
                if d.get("type") in ("TEXTURE", "COMPRESSED_TEXTURE")]
        by_off = {int(d["offset"], 16): d.get("symbol", n) for n, d in ents}
        for name, d in ents:
            sym, fmt = d.get("symbol", name), d["format"]
            comp = d["type"] == "COMPRESSED_TEXTURE"
            rel = f"src/assets/{VER}/{base}/{name}.{fmt.lower()}.{'incbin' if comp else 'inc'}.c"
            p = os.path.join(tree, rel)
            if not os.path.exists(p):
                missing.append(rel)
                continue
            mio0_bytes = 0
            if comp:
                raw = inc_bytes(p, 1)
                mio0_bytes = len(raw)
                data = crunch64.mio0.decompress(raw)
            else:
                data = inc_bytes(p, CSIZE[d.get("ctype", "u8")])
            e = {"file": rel, "yaml": base, "fmt": fmt, "ctype": d.get("ctype", "u8"), "bytes": len(data),
                 "mio0": comp, "_data": data}
            if comp:
                e["mio0_bytes"] = mio0_bytes
            if fmt == "TLUT":
                e["colors"] = int(d["colors"], 0)
            else:
                e["w"], e["h"] = int(d["width"], 0), int(d["height"], 0)
                if "tlut" in d and fmt.startswith("CI"):
                    e["tlut"] = by_off.get(int(d["tlut"], 16) & 0xFFFFFF)
            tex[sym] = e
    return tex, missing


def decode_all(tex, bad):
    """-> {symbol: rgba} for every non-TLUT texture."""
    for e in tex.values():
        if e["fmt"] == "TLUT":
            e["_pal"] = texfmt.decode(e["_data"], e["colors"], 1, texfmt.RGBA, texfmt.B16).reshape(-1, 4)
    out = {}
    for sym, e in tex.items():
        if e["fmt"] == "TLUT":
            continue
        f, s = FMTS[e["fmt"]]
        pal = None
        if f == texfmt.CI:
            t = tex.get(e.get("tlut") or "")
            if t is None or "_pal" not in t:
                bad.append(f"{sym} (tlut {e.get('tlut')} missing)")
                pal = np.stack([np.arange(256)] * 3 + [np.full(256, 255)], -1)
            else:
                pal = t["_pal"]
                need = 16 if s == texfmt.B4 else 256
                if len(pal) < need:
                    pal = np.concatenate([pal, np.zeros((need - len(pal), 4), np.uint8)])
        need = texfmt.texel_bytes(e["w"], e["h"], s)
        if len(e["_data"]) < need:
            bad.append(f"{sym} ({len(e['_data'])} < {need} bytes)")
            continue
        out[sym] = texfmt.decode(e["_data"], e["w"], e["h"], f, s, pal)
    return out


def main(argv):
    tree = argv[0]
    png_dir = argv[argv.index("--png") + 1] if "--png" in argv else None
    tex, missing = load_entries(tree)
    bad = []
    rgbas = decode_all(tex, bad)
    n_png = 0
    for sym, rgba in rgbas.items():
        e = tex[sym]
        fmt_name = e["fmt"]
        e.update(texture_fact(e["file"], rgba))
        e["fmt"] = fmt_name
        if png_dir:
            op = os.path.join(png_dir, e["yaml"], sym + ".png")
            os.makedirs(os.path.dirname(op), exist_ok=True)
            png.write(op, rgba)
            n_png += 1
    for e in tex.values():
        e.pop("_data", None)
        e.pop("_pal", None)
    for sym, e in tex.items():
        if e["fmt"] == "TLUT":
            e["users"] = sorted(s for s, t in tex.items() if t.get("tlut") == sym)
    os.makedirs(SPEC, exist_ok=True)
    json.dump(tex, open(os.path.join(SPEC, "textures.json"), "w"), indent=0, sort_keys=True)
    kinds = {}
    for e in tex.values():
        kinds[e["fmt"]] = kinds.get(e["fmt"], 0) + 1
    print(f"textures: {len(tex)} {kinds}; mio0: {sum(e['mio0'] for e in tex.values())}")
    print(f"missing: {len(missing)} {missing[:3]}")
    print(f"problems: {len(bad)} {bad[:5]}")
    print(f"TLUTs without users: {sum(1 for e in tex.values() if e['fmt'] == 'TLUT' and not e['users'])}")
    from . import masks
    masks.spec(tree)
    if png_dir:
        print(f"dirty pngs: {n_png} -> {png_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
