"""Course-select venue thumbnails (aVenue<Name>Tex, 64x32): a tiny scene per venue, drawn by us.

Sky and ground colours come from the kept 4x4 grid (top row = sky, bottom row = ground); the scene
elements (skyline, sea, dunes, trees, peaks, station, canyon, lava, void) are ours.
"""
import numpy as np
from PIL import Image, ImageDraw

SS = 8


def rows(e):
    g = np.asarray(e["grid"], np.float32).reshape(4, 4, 4)[..., :3]
    return g[0].mean(0), g[1].mean(0), g[2].mean(0), g[3].mean(0)


def c(v, k=1.0):
    return tuple(int(max(0, min(255, x * k))) for x in v) + (255,)


def scene(e, kind):
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    top, mid1, mid2, bot = rows(e)
    im = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / H
        col = top * (1 - t) + mid1 * t if t < 0.5 else mid1 * (2 - 2 * t) + bot * (2 * t - 1)
        d.line([(0, y), (W, y)], fill=c(col))
    rng = np.random.default_rng(len(kind))
    hz = H * 0.62
    if kind == "MuteCity":
        for i in range(14):
            x0 = i * W / 14
            bh = rng.uniform(0.25, 0.6) * H
            d.rectangle([x0, hz - bh, x0 + W / 16, H], fill=c(mid2, 0.45))
            for yy in np.arange(hz - bh + SS * 2, H, SS * 3):
                for xx in np.arange(x0 + SS, x0 + W / 16 - SS, SS * 3):
                    if rng.random() < 0.5:
                        d.rectangle([xx, yy, xx + SS, yy + SS], fill=(250, 230, 140, 255))
    elif kind == "PortTown":
        d.rectangle([0, hz, W, H], fill=c((40, 90, 150)))
        for i in range(4):
            x0 = rng.uniform(0, W)
            d.rectangle([x0, hz - H * 0.25, x0 + SS * 3, hz], fill=c(mid2, 0.5))
            d.line([(x0, hz - H * 0.25), (x0 + W * 0.12, hz - H * 0.35)], fill=c(mid2, 0.5), width=SS)
    elif kind == "BigBlue":
        d.rectangle([0, hz * 0.9, W, H], fill=c((30, 90, 200)))
        for i in range(10):
            y = hz + rng.uniform(0, H - hz)
            x = rng.uniform(0, W)
            d.arc([x, y, x + W * 0.1, y + SS * 3], 180, 360, fill=(220, 240, 255, 255), width=SS)
    elif kind == "SandOcean":
        for i in range(3):
            y0 = hz + i * H * 0.1
            d.chord([-W * 0.2 + i * W * 0.3, y0, W * 0.6 + i * W * 0.3, y0 + H * 0.6], 180, 360,
                    fill=c((230, 190, 110), 0.9 - i * 0.08))
        d.ellipse([W * 0.75, H * 0.08, W * 0.88, H * 0.34], fill=(255, 240, 180, 255))
    elif kind == "DevilsForest":
        for i in range(12):
            x = i * W / 11 + rng.uniform(-SS, SS)
            th = rng.uniform(0.35, 0.6) * H
            d.polygon([(x, H - th - H * 0.1), (x - W * 0.06, H), (x + W * 0.06, H)], fill=c((20, 70, 30)))
    elif kind == "WhiteLand":
        for i in range(4):
            x = i * W / 3
            d.polygon([(x - W * 0.25, H), (x, H * 0.25), (x + W * 0.25, H)], fill=c((190, 205, 230)))
            d.polygon([(x - W * 0.07, H * 0.45), (x, H * 0.25), (x + W * 0.07, H * 0.45)], fill=(250, 252, 255, 255))
    elif kind == "Sector":
        for i in range(30):
            x, y = rng.uniform(0, W), rng.uniform(0, hz)
            d.ellipse([x, y, x + SS, y + SS], fill=(240, 240, 255, 255))
        d.rectangle([W * 0.2, hz - H * 0.15, W * 0.8, hz], fill=c((120, 140, 170)))
        d.ellipse([W * 0.42, hz - H * 0.35, W * 0.58, hz - H * 0.05], fill=c((160, 180, 210)))
    elif kind == "RedCanyon":
        for i in range(3):
            x0 = i * W * 0.4 - W * 0.1
            d.polygon([(x0, H), (x0 + W * 0.05, H * 0.35), (x0 + W * 0.25, H * 0.3), (x0 + W * 0.32, H)],
                      fill=c((180, 70, 40), 0.8 + i * 0.1))
    elif kind == "FireField":
        d.rectangle([0, hz, W, H], fill=(60, 20, 10, 255))
        for i in range(12):
            x, y = rng.uniform(0, W), rng.uniform(hz, H)
            d.ellipse([x, y, x + W * 0.12, y + SS * 3], fill=(250, 120 + int(rng.uniform(0, 100)), 20, 255))
    elif kind == "Silence":
        d.rectangle([0, 0, W, H], fill=(12, 12, 20, 255))
        for i in range(8):
            y = hz + i * SS * 2
            d.line([(0, y), (W, y)], fill=(60, 60, 90, 255), width=SS // 2)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.uint8).copy()


def hook(sym, e):
    if sym.startswith("aVenue") and sym.endswith("Tex") and "w" in e:
        return scene(e, sym[6:-3])
    return None
