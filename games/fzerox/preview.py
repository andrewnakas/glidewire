"""Render clean pictures (hooks + grid default) for symbols matching a regex into a labelled sheet.

    python -m games.fzerox.preview <regex> <out.png> [--scale 2] [--width 1600] [--dirty]
--dirty puts the dirty picture beside each clean one (dev only; the sheet is never committed).
"""
import argparse
import json
import os
import re
import shutil
import tempfile

from cleanroom.gfx import png

from . import drawn, generate

DIRTY_PNG = "D:/n64work/fzerox/dirty_png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regex")
    ap.add_argument("out")
    ap.add_argument("--scale", default="2")
    ap.add_argument("--width", default="1600")
    ap.add_argument("--dirty", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(os.path.join(generate.SPEC, "textures.json"), encoding="utf-8"))
    generate.HOOKS[:] = drawn.HOOKS
    rx = re.compile(a.regex)
    tmp = tempfile.mkdtemp()
    n = 0
    for sym, e in spec.items():
        if e["fmt"] == "TLUT" or "grid" not in e or not rx.search(sym):
            continue
        os.makedirs(os.path.join(tmp, "c"), exist_ok=True)
        png.write(os.path.join(tmp, "c", sym + ".png"), generate.picture(sym, e))
        if a.dirty:
            src = os.path.join(DIRTY_PNG, e["yaml"], sym + ".png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(tmp, "c", sym + "~r.png"))
        n += 1
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "games.fzerox.sheet", tmp, a.out, ".", "--scale", a.scale, "--width", a.width])
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
