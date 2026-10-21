"""Menu pictures with lettering, drawn by us: cup shields, the mode-select signs, the title logo, credit cards.

Shapes and colours come from Glidewire's own kit (theme.py: shields, palettes; 2026-10 rebuild, no retail
outline or grid); every letter, emblem and figure is our own drawing (PIL at 8x supersampling, OFL fonts).
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cleanroom.decomp.gen import unpack_alpha2

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
SS = 8


def font(kind, px):
    fn, w = {"heavy": ("RussoOne-Regular.ttf", None), "italic": ("Exo2-Italic[wght].ttf", 900),
             "condensed": ("Teko[wght].ttf", 700), "bold": ("Rubik.ttf", 800)}[kind]
    f = ImageFont.truetype(os.path.join(FONTS, fn), max(4, int(px)))
    if w:
        f.set_variation_by_axes([w])
    return f


def grid(e):
    g = np.asarray(e["grid"], np.float32)
    n = int(round(len(g) ** 0.5))
    g = g.reshape(n, n, 4)
    a = g[..., 3:4] / 255.0
    return np.clip(g[..., :3] / np.maximum(a, 1e-3), 0, 255), g[..., 3]


def alpha(e):
    return unpack_alpha2(e["alpha2"], e["w"], e["h"]) if "alpha2" in e else np.full((e["h"], e["w"]), 255.0)


def fit_text(d, text, kind, box, fill, stroke=0, stroke_fill=(0, 0, 0), anchor="mm", stretch=True):
    """Largest font size for `text` inside box (x0, y0, x1, y1) in supersampled pixels; draws it."""
    x0, y0, x1, y1 = box
    lo, hi, best = 4, int((y1 - y0) * 1.6) + 8, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = font(kind, mid)
        b = d.textbbox((0, 0), text, font=f, stroke_width=stroke)
        if b[2] - b[0] <= x1 - x0 and b[3] - b[1] <= y1 - y0:
            best, lo = (f, b), mid + 1
        else:
            hi = mid - 1
    if best is None:
        return
    f, b = best
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.text((cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2), text, font=f, fill=fill, stroke_width=stroke,
           stroke_fill=stroke_fill)


def down(img, w, h):
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


# ------------------------------------------------------------------ cup shields

CUPS = {"Jack": ("SPARK", (240, 140, 30)), "Queen": ("VOLT", (40, 190, 210)), "King": ("SURGE", (60, 80, 200)),
        "Joker": ("STORM", (110, 60, 150)), "X": ("W", (24, 24, 30)), "Edit": ("MAKER", (230, 230, 236))}


def shield(e, key):
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    a = Image.fromarray(np.clip(alpha(e), 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    a = a.point(lambda v: 255 if v > 110 else 0)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if key == "QuestionMark":
        d.rectangle([0, 0, W, H], fill=(28, 30, 110, 255))
        fit_text(d, "?", "heavy", (W * 0.15, H * 0.08, W * 0.85, H * 0.8), (245, 245, 250, 255))
    else:
        name, col = CUPS[key]
        light = key == "Edit"
        d.rectangle([0, 0, W, H], fill=col + (255,))
        # upper panel: black with the series name, then the cup name, then "C U P"
        d.rectangle([0, 0, W, H * 0.62], fill=(250, 250, 250, 255) if light else (16, 16, 20, 255))
        ink = (20, 20, 24, 255) if light else (245, 245, 245, 255)
        if key == "Edit":
            fit_text(d, "MAKER", "heavy", (W * 0.12, H * 0.08, W * 0.88, H * 0.55), ink)
        else:
            fit_text(d, "GLIDEWIRE", "heavy", (W * 0.1, H * 0.05, W * 0.9, H * 0.2), ink)
            if key == "X":
                fit_text(d, "W", "heavy", (W * 0.25, H * 0.2, W * 0.75, H * 0.6), (40, 220, 240, 255), SS,
                         (250, 250, 250, 255))
            else:
                fit_text(d, name, "condensed", (W * 0.1, H * 0.2, W * 0.9, H * 0.6), ink)
        cup_ink = (250, 225, 40, 255) if key != "Edit" else (230, 190, 20, 255)
        fit_text(d, "CUP", "heavy", (W * 0.28, H * 0.63, W * 0.72, H * 0.76), cup_ink, SS // 2, (10, 10, 10))
    # rim along the silhouette
    rim = a.filter(ImageFilter.MaxFilter(3 * SS // 2 * 2 + 1))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste((235, 235, 240, 255), (0, 0), rim)
    out.paste(img, (0, 0), a)
    return down(out, w, h)


# ------------------------------------------------------------------ mode signs

SIGNS = {"GpRace": ["CIRCUIT"], "TimeAttack": ["TIME", "TRIAL"], "DeathRace": ["KNOCK", "OUT"],
         "VsBattle": ["DUEL"], "Practice": ["WARM", "UP"], "Options": ["SETUP"],
         "CourseEdit": ["TRACK", "MAKER"], "CreateMachine": ["CRAFT", "MAKER"]}


def mode_sign(e, key):
    """A sign board (black, coloured letters from the grid) held up by our small pit-crew mascot."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    rgb, _ = grid(e)
    n = len(rgb)

    def vivid(rows):
        c = rgb[rows].reshape(-1, 3)
        sat = c.max(1) - c.min(1) + 0.3 * c.max(1)
        c = c[sat.argmax()]
        return c * (245.0 / max(c.max(), 1.0))

    top_c = vivid(slice(0, max(1, n // 4)))
    bot_c = vivid(slice(n // 4, n // 2))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bx0, by0, bx1, by1 = W * 0.16, H * 0.02, W * 0.84, H * 0.62
    d.rectangle([bx0, by0, bx1, by1], fill=(235, 235, 235, 255))
    d.rectangle([bx0 + SS, by0 + SS, bx1 - SS, by1 - SS], fill=(12, 12, 16, 255))
    lines = SIGNS[key]
    if len(lines) == 1:
        fit_text(d, lines[0], "condensed", (bx0 + 2 * SS, by0 + 4 * SS, bx1 - 2 * SS, by1 - 4 * SS),
                 tuple(int(v) for v in top_c) + (255,))
    else:
        mid = by0 + (by1 - by0) * 0.52
        fit_text(d, lines[0], "condensed", (bx0 + 2 * SS, by0 + 2 * SS, bx1 - 2 * SS, mid),
                 tuple(int(v) for v in top_c) + (255,))
        fit_text(d, lines[1], "condensed", (bx0 + 2 * SS, mid, bx1 - 2 * SS, by1 - 2 * SS),
                 tuple(int(v) for v in bot_c) + (255,))
    # mascot: round head with a helmet, hands gripping the sign's lower corners
    skin, suit, helm = (250, 200, 160, 255), (230, 60, 150, 255), (250, 205, 40, 255)
    cx = W / 2
    d.ellipse([cx - W * 0.36, H * 0.74, cx + W * 0.36, H * 1.1], fill=suit, outline=(40, 20, 40, 255), width=SS)
    hr = W * 0.17
    hy = H * 0.73
    d.ellipse([cx - hr, hy - hr, cx + hr, hy + hr], fill=skin, outline=(40, 20, 40, 255), width=SS)
    d.chord([cx - hr * 1.08, hy - hr * 1.12, cx + hr * 1.08, hy + hr * 0.5], 180, 360, fill=helm,
            outline=(40, 20, 40, 255), width=SS)
    for sx in (-1, 1):
        ex = cx + sx * hr * 0.4
        d.ellipse([ex - hr * 0.16, hy + hr * 0.05, ex + hr * 0.16, hy + hr * 0.42], fill=(30, 30, 40, 255))
        hx = cx + sx * W * 0.3
        d.line([(cx + sx * W * 0.25, H * 0.86), (hx, by1)], fill=suit, width=4 * SS)
        d.ellipse([hx - 3 * SS, by1 - 3 * SS, hx + 3 * SS, by1 + 3 * SS], fill=(245, 245, 245, 255),
                  outline=(40, 20, 40, 255), width=SS // 2)
    d.arc([cx - hr * 0.4, hy + hr * 0.2, cx + hr * 0.4, hy + hr * 0.75], 20, 160, fill=(120, 30, 40, 255), width=SS)
    return down(img, w, h)


# ------------------------------------------------------------------ title logo, cards

def title_logo(e):
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # GLIDEWIRE: chrome italic word riding on a glowing cyan wire with an orange spark at its tip
    for col, wid in (((20, 110, 140, 255), 7), ((60, 230, 250, 255), 3)):
        d.line([(W * 0.02, H * 0.86), (W * 0.55, H * 0.8), (W * 0.98, H * 0.66)], fill=col, width=wid * SS)
    d.ellipse([W * 0.93, H * 0.6, W * 0.99, H * 0.72], fill=(255, 150, 40, 255))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    fit_text(ld, "GLIDEWIRE", "italic", (W * 0.02, H * 0.08, W * 0.98, H * 0.72), (255, 255, 255, 255), 2 * SS,
             (10, 40, 60))
    arr = np.asarray(layer).astype(np.float32)
    t = np.linspace(0, 1, H)[:, None]
    chrome = np.stack([200 - 90 * t + 40 * np.sin(t * 9), 215 - 70 * t + 30 * np.sin(t * 9), 255 - 40 * t], -1)
    white = (arr[..., 0] > 200) & (arr[..., 2] > 200)
    arr[..., :3] = np.where(white[..., None], np.broadcast_to(chrome, arr[..., :3].shape), arr[..., :3])
    layer = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.alpha_composite(layer)
    return down(img, w, h)


def see_you(e):
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(["RIDE", "THE", "WIRE!"]):
        fit_text(d, t, "heavy", (W * 0.06, H * (0.04 + i * 0.31), W * 0.94, H * (0.33 + i * 0.31)),
                 (120, 250, 150, 255))
    return down(img, w, h)


def hook(sym, e):
    if "w" not in e:
        return None
    if sym.startswith("aCupSelect") and sym.endswith("Tex"):
        return shield(e, sym[10:-3])
    if sym.startswith("aMenuSign") and sym[9:-3] in SIGNS:
        return mode_sign(e, sym[9:-3])
    if sym == "aTitleLogoTex":
        return title_logo(e)
    if sym == "aCreditsSeeYouAgainTex":
        return see_you(e)
    return None
