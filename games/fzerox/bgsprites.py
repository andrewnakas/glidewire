"""Background panorama sprites (the decomp's BG_SPRITE_* table in ovl_i3/background.c): our own silhouette per
sprite kind (silhouette(); 2026-10 rebuild, no retail outline) and our own painting inside it.

  day skylines / mountains / towers / statues   pale stone shaded from the silhouette's distance field (light from
                                                the upper left), faint window rows on buildings
  NIGHT_* skylines, cup and sign buildings      dark mass with lit windows; the sign buildings carry our lettering
  MOON_*                                        a shaded sphere with craters
Colours follow the kept grid; windows, shading, craters and signs are ours.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from cleanroom.decomp.gen import unpack_alpha2

from .signs import fit_text

SPRITES = {"D_F220810": "CITY_SKYLINE_1", "D_F221210": "CITY_SKYLINE_2", "D_F221C10": "CITY_SKYLINE_3",
           "D_F222610": "CITY_SKYLINE_4", "D_F223010": "MOUNTAINS_1", "D_F223A10": "SKULL_MOUNTAIN",
           "D_F224410": "GIANT_TREE", "D_F224E10": "MOUNTAIN_CITY", "D_F225810": "CITY_SKYLINE_5",
           "D_F226210": "CITY_SKYLINE_6", "D_F226C10": "CITY_SKYLINE_7", "D_F227610": "CITY_SKYLINE_8",
           "D_F228010": "NIGHT_CITY_SKYLINE_5", "D_F228A10": "NIGHT_CITY_SKYLINE_6", "D_F229410": "NIGHT_CITY_SKYLINE_7",
           "D_F229E10": "NIGHT_CITY_SKYLINE_8", "D_F22A810": "MOUNTAINS_2", "D_F22B210": "MOUNTAINS_3",
           "D_F22BC10": "MOUNTAINS_TOWERS_1", "D_F22C610": "MOUNTAINS_TOWERS_2", "D_F22D010": "TOWERS_1",
           "D_F22DA10": "PYRAMID", "D_F22E410": "TOWERS_2", "D_F22EE10": "MOON_1", "D_F22F810": "STATUE_1",
           "D_F230210": "MOUNTAINS_TOWERS_3", "D_F230C10": "CASTLE", "D_F231610": "MOUNTAINS_TOWERS_4",
           "D_F232010": "ROCKET_LAUNCH_1", "D_F232A10": "ROCKET_LAUNCH_2", "D_F233410": "ROCKET_LAUNCH_3",
           "D_F233E10": "RADAR_DISH", "D_F234810": "RAISED_CITY_1", "D_F235210": "RAISED_GLASS_DOME",
           "D_F235C10": "RAISED_CITY_2", "D_F236610": "CONNECTED_MOUNTAINS", "D_F237010": "MOON_2",
           "D_F237A10": "FLYING_CITY", "D_F238410": "NIGHT_CITY_SKYLINE_1", "D_F238E10": "NIGHT_CITY_SKYLINE_2",
           "D_F239810": "NIGHT_CITY_SKYLINE_3", "D_F23A210": "NIGHT_CITY_SKYLINE_4", "D_F23AC10": "NINTENDO_BUILDING",
           "D_F23B610": "JACK_CUP_BUILDING", "D_F23C010": "NINTENDO_N_BUILDING", "D_F23CA10": "FZERO_X_BUILDING",
           "D_F23D410": "DOUBLE_SPHEROID_BUILDING", "D_F23DE10": "QUEEN_CUP_BUILDING",
           "D_F23E810": "KING_CUP_BUILDING", "D_F23F210": "JOKER_CUP_BUILDING", "D_F23FC10": "X_CUP_BUILDING",
           "D_F240610": "EDIT_CUP_BUILDING", "D_F241010": "MAN_STATUE_2", "D_F241A10": "MOON_3",
           "D_F242410": "SINGLE_SPHEROID_BUILDING"}
SIGN = {"JACK_CUP_BUILDING": "SPARK", "QUEEN_CUP_BUILDING": "VOLT", "KING_CUP_BUILDING": "SURGE",
        "JOKER_CUP_BUILDING": "STORM", "X_CUP_BUILDING": "WILD", "EDIT_CUP_BUILDING": "MAKER",
        "FZERO_X_BUILDING": "GLIDEWIRE", "NINTENDO_BUILDING": "GW", "NINTENDO_N_BUILDING": "G"}


def silhouette(kind, w, h, rng):
    """Our own skyline shape for a background sprite kind (2026-10 rebuild: no retail outline)."""
    S = 4
    im = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(im)
    W, H = w * S, h * S
    if kind.startswith("MOON"):
        r = min(W, H) * 0.42
        d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=255)
    elif "MOUNTAIN" in kind or kind == "SKULL_MOUNTAIN":
        xs = np.linspace(0, W, 24)
        ys = H * (0.35 + 0.45 * np.abs(np.cumsum(rng.uniform(-0.25, 0.25, 24))) % 0.6)
        d.polygon([(0, H)] + list(zip(xs, ys)) + [(W, H)], fill=255)
        if "TOWERS" in kind:
            for _ in range(3):
                x = rng.uniform(0.1, 0.9) * W
                d.rectangle([x, H * rng.uniform(0.05, 0.3), x + W * 0.04, H], fill=255)
    elif kind == "PYRAMID":
        d.polygon([(W * 0.1, H), (W * 0.5, H * 0.1), (W * 0.9, H)], fill=255)
    elif "STATUE" in kind:
        d.rectangle([W * 0.35, H * 0.8, W * 0.65, H], fill=255)
        d.ellipse([W * 0.42, H * 0.1, W * 0.58, H * 0.28], fill=255)
        d.polygon([(W * 0.38, H * 0.3), (W * 0.62, H * 0.3), (W * 0.58, H * 0.8), (W * 0.42, H * 0.8)], fill=255)
        d.line([(W * 0.6, H * 0.35), (W * 0.8, H * 0.08)], fill=255, width=S * 2)
    elif kind == "GIANT_TREE":
        d.rectangle([W * 0.45, H * 0.5, W * 0.55, H], fill=255)
        for _ in range(6):
            cx, cy, r = rng.uniform(0.3, 0.7) * W, rng.uniform(0.15, 0.5) * H, rng.uniform(0.12, 0.22) * W
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif kind.startswith("ROCKET"):
        d.rectangle([W * 0.44, H * 0.15, W * 0.56, H * 0.85], fill=255)
        d.polygon([(W * 0.44, H * 0.15), (W * 0.5, H * 0.02), (W * 0.56, H * 0.15)], fill=255)
        d.rectangle([W * 0.2, H * 0.85, W * 0.8, H], fill=255)
    elif kind == "RADAR_DISH":
        d.rectangle([W * 0.46, H * 0.45, W * 0.54, H], fill=255)
        d.chord([W * 0.15, H * 0.05, W * 0.85, H * 0.65], 200, 340, fill=255)
    elif "DOME" in kind or "SPHEROID" in kind:
        d.rectangle([W * 0.1, H * 0.7, W * 0.9, H], fill=255)
        d.ellipse([W * 0.2, H * 0.25, W * 0.8, H * 0.95], fill=255)
        if "DOUBLE" in kind:
            d.ellipse([W * 0.55, H * 0.05, W * 0.9, H * 0.45], fill=255)
            d.rectangle([W * 0.7, H * 0.4, W * 0.74, H * 0.7], fill=255)
    elif kind == "CASTLE":
        d.rectangle([W * 0.1, H * 0.4, W * 0.9, H], fill=255)
        for x in np.linspace(0.1, 0.84, 7):
            d.rectangle([W * x, H * 0.32, W * (x + 0.06), H * 0.4], fill=255)
        d.rectangle([W * 0.42, H * 0.1, W * 0.58, H * 0.4], fill=255)
    elif kind == "FLYING_CITY" or kind.startswith("RAISED"):
        d.polygon([(W * 0.1, H * 0.6), (W * 0.9, H * 0.6), (W * 0.5, H * 0.98)], fill=255)
        for _ in range(7):
            x = rng.uniform(0.15, 0.8) * W
            d.rectangle([x, H * rng.uniform(0.1, 0.5), x + W * rng.uniform(0.05, 0.1), H * 0.62], fill=255)
    elif kind.endswith("_BUILDING"):
        d.rectangle([W * 0.2, H * 0.12, W * 0.8, H], fill=255)
        d.rectangle([W * 0.1, H * 0.6, W * 0.9, H], fill=255)
    else:   # city skylines: a row of our own blocks
        x = 0.0
        while x < W:
            bw = rng.uniform(0.05, 0.14) * W
            top = H * rng.uniform(0.1, 0.7)
            d.rectangle([x, top, x + bw, H], fill=255)
            if rng.random() < 0.3:
                d.rectangle([x + bw * 0.4, top - H * 0.12, x + bw * 0.55, top], fill=255)
            x += bw * rng.uniform(0.8, 1.1)
    return np.asarray(im.resize((w, h), Image.BILINEAR), np.float32) > 110


def base_colour(e):
    g = np.asarray(e["grid"], np.float32).reshape(-1, 4)
    ok = g[:, 3] > 40
    if not ok.any():
        return np.array([200.0, 200.0, 205.0])
    return (g[ok, :3] / (g[ok, 3:4] / 255.0)).clip(0, 255).mean(0)


def paint(sym, e):
    kind = SPRITES[sym]
    w, h = e["w"], e["h"]
    rng = np.random.default_rng(len(kind) * 7 + w)
    m = silhouette(kind, w, h, rng)
    if m.sum() < 10:
        return None
    col = base_colour(e)
    d = ndimage.distance_transform_edt(m)
    hgt = ndimage.gaussian_filter(np.sqrt(d), 1.2)
    gy, gx = np.gradient(hgt)
    shade = np.clip(0.75 - 0.9 * gx - 0.7 * gy, 0.45, 1.15)
    night = kind.startswith("NIGHT") or kind.endswith("_BUILDING")
    if kind.startswith("MOON"):
        cy, cx = np.nonzero(m)
        mx, my, r = cx.mean(), cy.mean(), max(1.0, np.sqrt(m.sum() / np.pi))
        yy, xx = np.mgrid[0:h, 0:w]
        nx, ny = (xx - mx) / r, (yy - my) / r
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
        shade = np.clip(0.25 + 0.85 * (-0.5 * nx - 0.5 * ny + 0.7 * nz), 0.2, 1.2)
        rgb = col[None, None] * shade[..., None]
        for _ in range(9):
            qx, qy, qr = rng.uniform(mx - r * 0.7, mx + r * 0.7), rng.uniform(my - r * 0.7, my + r * 0.7), \
                rng.uniform(r * 0.08, r * 0.2)
            cr = ((xx - qx) ** 2 + (yy - qy) ** 2) < qr * qr
            rgb[cr] *= 0.78
    elif night:
        rgb = np.broadcast_to(np.array([34.0, 32.0, 46.0]) + col * 0.1, (h, w, 3)).copy() * shade[..., None]
        win = np.zeros((h, w), bool)
        for y in range(2, h, 3):
            for x in range(1, w, 2):
                if m[y, x] and d[y, x] > 1.5 and rng.random() < 0.35:
                    win[y, x] = True
        palette = np.array([[250, 220, 120], [255, 170, 60], [140, 220, 255], [250, 250, 230]], np.float32)
        idx = rng.integers(0, len(palette), (h, w))
        rgb[win] = palette[idx[win]]
    else:
        rgb = col[None, None] * shade[..., None]
        if any(k in kind for k in ("CITY", "TOWERS", "ROCKET", "RAISED", "SPHEROID", "CASTLE")):
            for y in range(3, h, 4):
                rgb[y, :] *= np.where(m[y] & (d[y] > 1.5), 0.86, 1.0)[..., None]
    rgb[m & (d < 1.01)] *= 0.7            # a darker rim
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = np.where(m, 255, 0)
    img = out.astype(np.uint8)
    if kind in SIGN:
        img = sign(img, m, SIGN[kind])
    return img


def sign(img, m, text):
    """A lit sign panel across the top of the building mass."""
    h, w = m.shape
    rows = np.nonzero(m.any(1))[0]
    top = rows[0]
    cols = np.nonzero(m[min(h - 1, top + 4)])[0]
    if len(cols) < 4:
        return img
    x0, x1 = cols[0] + 1, cols[-1]
    y0, y1 = top + 2, min(h - 1, top + 2 + max(6, (x1 - x0) // 4))
    SS = 8
    im = Image.fromarray(img).resize((w * SS, h * SS), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=(16, 20, 40, 255), outline=(120, 240, 255, 255), width=SS)
    fit_text(d, text, "heavy", (x0 * SS + SS, y0 * SS + SS, x1 * SS - SS, y1 * SS - SS), (250, 250, 255, 255))
    out = np.asarray(im.resize((w, h), Image.LANCZOS), np.uint8).copy()
    out[..., 3] = img[..., 3]
    return out


def hook(sym, e):
    if sym in SPRITES and "w" in e:
        return paint(sym, e)
    return None
