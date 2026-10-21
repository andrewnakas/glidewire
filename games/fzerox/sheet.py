"""DIRTY ROOM (or clean, with --clean): labelled contact sheet of textures by symbol regex.

    python -m games.fzerox.sheet <png dir> <out.png> <regex> [--scale 2] [--width 1600] [--min-text 0]

<png dir> holds <ast>/<symbol>.png (dirty_png from extract_spec, or a clean preview dir).
Each tile gets its symbol name above it. Dirty sheets are for transcription only; never committed.
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from cleanroom.gfx import png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pngdir")
    ap.add_argument("out")
    ap.add_argument("regex")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--bg", default="40,40,60")
    a = ap.parse_args()
    rx = re.compile(a.regex)
    items = []
    for p in sorted(glob.glob(os.path.join(a.pngdir, "*", "*.png"))):
        sym = os.path.splitext(os.path.basename(p))[0]
        if rx.search(sym):
            items.append((sym, png.read(p)))
    bg = tuple(int(v) for v in a.bg.split(","))
    x = y = rowh = 0
    place = []
    for sym, im in items:
        s = a.scale
        h, w = im.shape[0] * s, im.shape[1] * s
        tw = max(w, len(sym) * 6 + 4)
        if x and x + tw > a.width:
            x, y, rowh = 0, y + rowh + 16, 0
        place.append((sym, im, x, y + 12, s))
        x += tw + 8
        rowh = max(rowh, h + 12)
    H = y + rowh + 16
    sheet = Image.new("RGB", (a.width, H), bg)
    d = ImageDraw.Draw(sheet)
    for sym, im, x0, y0, s in place:
        big = Image.fromarray(np.repeat(np.repeat(im, s, 0), s, 1), "RGBA")
        sheet.paste(big, (x0, y0), big)
        d.text((x0, y0 - 11), sym, fill=(255, 255, 0))
    sheet.save(a.out)
    print(f"{len(items)} tiles -> {a.out} ({a.width}x{H})")


if __name__ == "__main__":
    main()
