"""Re-typeset text-bearing textures (games/fzerox/text_labels.json) with OFL fonts.

Styles: menu (bold italic), header (heavy italic, vertical gradient), block (upright heavy caps),
box (white on the black panel with a frame, left aligned), plain (upright bold).
Colours come from the kept colour grid: the mean of the opaque cells (header: top and bottom rows of them).
I4/IA textures are drawn white (the game tints them). Fonts in games/fzerox/fonts (see README.md).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
LABELS = os.path.join(HERE, "text_labels.json")
SS = 4

FACES = {
    "menu": ("Exo2-Italic[wght].ttf", 700),
    "header": ("Exo2-Italic[wght].ttf", 900),
    "block": ("RussoOne-Regular.ttf", None),
    "box": ("Rubik.ttf", 500),
    "plain": ("Rubik.ttf", 700),
}


def font(style, px):
    fn, var = FACES[style]
    f = ImageFont.truetype(os.path.join(FONTS, fn), max(4, int(px)))
    if var:
        try:
            f.set_variation_by_axes([var])
        except Exception:
            pass
    return f


def grid_colours(e):
    """-> (mean ink rgb, top rgb, bottom rgb) from the opaque cells of the kept grid."""
    g = np.asarray(e["grid"], np.float32)
    n = int(round(len(g) ** 0.5))
    g = g.reshape(n, n, 4)
    a = g[..., 3:4] / 255.0
    ok = a[..., 0] > 0.15
    if not ok.any():
        w = np.array([255.0, 255.0, 255.0])
        return w, w, w
    rgb = np.clip(g[..., :3] / np.maximum(a, 1e-3), 0, 255)
    # ink = the brighter opaque cells (the retail letters carry a dark rim that darkens the plain mean)
    cells = rgb[ok]
    lum = cells @ np.array([0.3, 0.59, 0.11])
    mean = cells[lum >= np.quantile(lum, 0.5)].mean(0)
    if mean.max() < 235:
        mean = mean * (235.0 / max(mean.max(), 1.0))
    rows = [r for r in range(n) if ok[r].any()]
    def bright(c):
        c = np.asarray(c)
        return c * (200.0 / max(float(c.max()), 1.0)) if c.max() < 200 else c

    top = bright(rgb[rows[0]][ok[rows[0]]].max(0))
    bot = bright(rgb[rows[-1]][ok[rows[-1]]].max(0))
    return mean, top, bot


def _fit(draw, lines, style, W, H, pad_x, pad_y):
    """Largest font size where every line fits; returns (font, line height, widths)."""
    lo, hi = 4, H * 2
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = font(style, mid)
        boxes = [draw.textbbox((0, 0), ln, font=f) for ln in lines]
        lh = max(b[3] - b[1] for b in boxes) if boxes else mid
        total = lh * len(lines) + (len(lines) - 1) * mid * 0.12
        wmax = max(b[2] - b[0] for b in boxes)
        if wmax <= W - 2 * pad_x and total <= H - 2 * pad_y:
            best = (f, lh, boxes, mid)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        f = font(style, 4)
        boxes = [draw.textbbox((0, 0), ln, font=f) for ln in lines]
        best = (f, max(b[3] - b[1] for b in boxes), boxes, 4)
    return best


def render(label, e):
    w, h = e["w"], e["h"]
    style = label.get("style", "menu")
    lines = label["text"].split("\n")
    W, H = w * SS, h * SS
    tint = e["fmt"].startswith(("I4", "I8", "IA"))
    mean, top, bot = grid_colours(e)
    if tint:
        mean = top = bot = np.array([255.0, 255.0, 255.0])
    if style == "box":
        img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], outline=(235, 235, 235, 255), width=SS)
        f, lh, boxes, px = _fit(d, lines, style, W, H, 3 * SS, 2 * SS)
        y = (H - (lh * len(lines) + (len(lines) - 1) * px * 0.12)) / 2
        for ln, b in zip(lines, boxes):
            d.text((4 * SS - b[0], y - b[1]), ln, font=f, fill=(255, 255, 255, 255))
            y += lh + px * 0.12
        out = img.resize((w, h), Image.LANCZOS)
        return np.asarray(out, np.uint8).copy()
    # text on transparency: a mask, then colour
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    pad = SS if h >= 12 else 0
    f, lh, boxes, px = _fit(d, lines, style, W, H, pad, pad // 2)
    y = (H - (lh * len(lines) + (len(lines) - 1) * px * 0.12)) / 2
    for ln, b in zip(lines, boxes):
        x = (W - (b[2] - b[0])) / 2 - b[0] if label.get("align", "center") == "center" else pad - b[0]
        d.text((x, y - b[1]), ln, font=f, fill=255)
        y += lh + px * 0.12
    m = np.asarray(mask.resize((w, h), Image.LANCZOS), np.float32) / 255.0
    rgb = np.zeros((h, w, 3), np.float32)
    if style == "header" and not tint:
        t = np.linspace(0, 1, h)[:, None, None]
        rgb[:] = top[None, None] * (1 - t) + bot[None, None] * t
    else:
        rgb[:] = mean
    if not tint and h >= 12:
        # a dark rim one texel wide so the letters read on any background
        from scipy.ndimage import maximum_filter
        rim = maximum_filter(m, size=3)
        edge = np.clip(rim - m, 0, 1)
        rgb = rgb * (1 - edge[..., None]) + np.array([20.0, 20.0, 40.0]) * edge[..., None]
        m = np.maximum(m, rim * 0.95)
    a = np.clip(m * 255.0 * 1.25, 0, 255)
    out = np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)
    return out


_LABELS = None


def hook(sym, e):
    global _LABELS
    if _LABELS is None:
        _LABELS = {k: v for k, v in json.load(open(LABELS, encoding="utf-8")).items() if not k.startswith("_")}
    lab = _LABELS.get(sym)
    if lab is None:
        return None
    return render(lab, e)
