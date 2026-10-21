"""Track-side pictures with meaning, drawn by us: the jump plate, pit-zone and surface tiles (dirt, ice), and the
roadside billboards (our own sponsors and a TV-show board).

Colours follow the kept grids; patterns, emblems and all lettering are ours. These live in the MIO0-compressed
course_track_gfx segment, so the patterns are kept simple and periodic.
"""
import numpy as np
from PIL import Image, ImageDraw

from .signs import fit_text

SS = 8


def canvas(e, bg):
    w, h = e["w"], e["h"]
    im = Image.new("RGBA", (w * SS, h * SS), tuple(int(v) for v in bg) + (255,))
    return im, ImageDraw.Draw(im), w * SS, h * SS


def done(im, e):
    return np.asarray(im.resize((e["w"], e["h"]), Image.LANCZOS), np.uint8).copy()


def jump_plate(e):
    im, d, W, H = canvas(e, (120, 124, 132))
    d.rectangle([SS, SS, W - SS, H - SS], outline=(70, 72, 80, 255), width=2 * SS)
    # a red diamond with a white up-chevron: "launch here"
    cx, cy, r = W / 2, H / 2, W * 0.34
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(215, 30, 45, 255),
              outline=(60, 10, 20, 255), width=SS)
    d.polygon([(cx, cy - r * 0.55), (cx + r * 0.45, cy + r * 0.05), (cx + r * 0.2, cy + r * 0.05),
               (cx, cy - r * 0.18), (cx - r * 0.2, cy + r * 0.05), (cx - r * 0.45, cy + r * 0.05)],
              fill=(250, 250, 250, 255))
    return done(im, e)


def pit_tile(e):
    """Pink energy honeycomb (the pit zone refills energy)."""
    im, d, W, H = canvas(e, (200, 90, 170))
    r = W / 4
    for row in range(-1, 5):
        for col in range(-1, 3):
            cx = col * r * 1.75 * 1.0 + (r * 0.875 if row % 2 else 0) + r
            cy = row * r * 1.0 + r * 0.5
            pts = [(cx + r * 0.8 * np.cos(a), cy + r * 0.55 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
            d.polygon(pts, fill=(245, 160, 230, 255), outline=(150, 40, 130, 255), width=SS)
    return done(im, e)


def noise_tile(e, base, var, seed, blobs=18):
    """Periodic blotchy surface (dirt, ice) in the grid's colour."""
    w, h = e["w"], e["h"]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), np.float32)
    for _ in range(blobs):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1.5, 4.5)
        dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx))
        dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy))
        img += rng.choice([-1, 1]) * np.exp(-(dx ** 2 + dy ** 2) / (2 * r * r))
    img = np.round(img * 2) / 2
    rgb = np.clip(np.asarray(base, np.float32)[None, None] + img[..., None] * np.asarray(var, np.float32), 0, 255)
    return np.dstack([rgb, np.full((h, w), 255.0)]).astype(np.uint8)


def grid_mean(e):
    g = np.asarray(e["grid"], np.float32).reshape(-1, 4)
    return g[:, :3].mean(0)


def billboard_tv(e):
    im, d, W, H = canvas(e, (20, 30, 70))
    d.rectangle([0, 0, W - 1, H - 1], outline=(220, 220, 230, 255), width=SS)
    fit_text(d, "GLIDE TV", "italic", (W * 0.06, H * 0.06, W * 0.94, H * 0.6), (250, 220, 40, 255), SS,
             (120, 20, 20))
    fit_text(d, "FRI 8:30 PM", "bold", (W * 0.2, H * 0.62, W * 0.8, H * 0.92), (240, 240, 250, 255))
    return done(im, e)


def billboard_band(e):
    im, d, W, H = canvas(e, (230, 120, 30))
    for i in range(6):
        d.rectangle([0, H * i / 6, W, H * (i + 0.5) / 6], fill=(240, 150, 40, 255))
    fit_text(d, "VOLT", "heavy", (W * 0.05, H * 0.05, W * 0.6, H * 0.55), (30, 20, 40, 255))
    fit_text(d, "LIVE!", "italic", (W * 0.4, H * 0.45, W * 0.95, H * 0.95), (250, 250, 250, 255), SS // 2,
             (30, 20, 40))
    return done(im, e)


def sponsor(e, text, bg, fg):
    im, d, W, H = canvas(e, bg)
    d.rectangle([SS, SS, W - SS, H - SS], outline=tuple(fg) + (255,), width=2 * SS)
    fit_text(d, text, "heavy", (W * 0.15, H * 0.15, W * 0.85, H * 0.85), tuple(fg) + (255,))
    return done(im, e)


def hook(sym, e):
    if "w" not in e:
        return None
    if sym == "aJumpFeatureTex":
        return jump_plate(e)
    if sym == "aPitEffectTex":
        return pit_tile(e)
    if sym == "aDirtEffectTex":
        return noise_tile(e, grid_mean(e), (26, 20, 14), 11)
    if sym == "aIceEffectTex":
        return noise_tile(e, grid_mean(e) * 0.9 + 25, (22, 26, 30), 12, 10)
    if sym == "D_801C018":
        return billboard_tv(e)
    if sym == "D_801B018":
        return billboard_band(e)
    if sym == "D_801E2A8":
        return sponsor(e, "Z", (230, 40, 40), (250, 220, 40))
    if sym == "D_801D960":
        return sponsor(e, "GO", (30, 80, 200), (250, 250, 250))
    return None
