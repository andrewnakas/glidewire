"""CLEAN ROOM: Glidewire's own texture kit. Every texture that no drawing hook paints is designed here from
its structural facts only (size, format, segment, whether it has transparency and roughly how much), the
symbol name and our own palettes. No retail colour grid or outline is read (2026-10 originality rebuild).

    texture(sym, e) -> RGBA uint8 (h, w, 4)       our design for one texture slot
    grid(sym, e, n) -> [[r, g, b, a], ...]          n x n colour summary of our design (for hooks that colour
                                                     their drawings from a grid)
    alpha2(sym, e) -> hex string                     our 2-bit coverage mask in the spec's packing (for hooks
                                                     that cut their drawings to a shape)

Patterns: panel plating, chevrons, hex cells, ribs, rings, dots, bands, brushed metal and soft terrain,
picked per slot by a seeded draw; colours from the venue palette of the segment. Shapes for transparent
textures: discs, rings, stars, bursts, puffs, wedges, figures and leaves (by name hints, else seeded).
"""
import numpy as np

from cleanroom.decomp.gen import h32

# our palettes (dark, mid, light, accent, accent2)
PAL = {
    "mute_city_textures": [(18, 24, 44), (40, 120, 150), (190, 235, 245), (240, 60, 160), (250, 200, 60)],   # Lumen City
    "port_town_textures": [(36, 26, 22), (150, 84, 40), (230, 190, 140), (60, 150, 170), (240, 230, 210)],  # Dockspire
    "big_blue_textures": [(10, 40, 70), (30, 140, 190), (210, 245, 250), (250, 250, 250), (255, 150, 80)],  # Tidewater
    "sand_ocean_textures": [(70, 40, 20), (190, 130, 60), (245, 215, 160), (120, 60, 120), (90, 170, 160)],  # Ochre Flats
    "devils_forest_textures": [(20, 30, 18), (70, 100, 40), (170, 190, 110), (120, 70, 40), (230, 120, 60)],  # Thornwood
    "white_land_textures": [(40, 50, 80), (150, 170, 220), (240, 245, 255), (180, 120, 220), (90, 200, 230)],  # Frostmere
    "sector_textures": [(24, 26, 28), (90, 100, 96), (200, 230, 120), (120, 240, 80), (250, 250, 250)],     # Vector Yard
    "red_canyon_textures": [(50, 20, 30), (160, 70, 50), (230, 160, 110), (110, 50, 110), (250, 210, 120)],  # Rust Gorge
    "fire_field_textures": [(20, 16, 16), (70, 60, 56), (250, 170, 60), (240, 80, 30), (255, 230, 150)],    # Cinder Plain
    "silence_textures": [(14, 14, 40), (70, 70, 140), (200, 200, 230), (120, 220, 250), (160, 120, 230)],   # Hush Valley
    "ending_venue_textures": [(40, 30, 10), (170, 140, 60), (250, 240, 200), (250, 250, 250), (90, 160, 240)],
    "_brand": [(22, 24, 32), (60, 80, 110), (220, 230, 240), (40, 220, 240), (255, 140, 40)],              # Glidewire
}
SHAPE_HINTS = [("cupselect", "shield"), ("font", "rect"), ("position", "rect"), ("digit", "rect"),
               ("number", "rect"), ("char", "rect"), ("letter", "rect"), ("result", "rect"), ("machinestat", "rect"),
               ("machineweight", "rect"), ("lap", "rect"), ("time", "rect"), ("menu", "rect"), ("namecard", "rect"),
               ("ending", "rect"), ("title", "rect"), ("options", "rect"), ("select", "rect"), ("portrait", "rect"),
               ("explosion", "burst"), ("spark", "star"), ("cloud", "puff"), ("smoke", "puff"), ("tree", "leaf"),
               ("star", "star"), ("ring", "ring"), ("shadow", "disc"), ("light", "disc"), ("flare", "star"),
               ("fullportrait", "figure"), ("credits", "wedge"), ("marker", "diamond"), ("arrow", "wedge"),
               ("trail", "streak"), ("flame", "leaf"), ("fire", "leaf")]
PATTERNS = ["panel", "chevron", "hex", "ribs", "rings", "dots", "bands", "brushed", "terrain", "grate"]


def _pal(e):
    return np.asarray(PAL.get(e.get("yaml"), PAL["_brand"]), np.float32)


def _noise(w, h, rng, cells, wrap=True):
    g = rng.uniform(0, 1, (cells + 1, cells + 1))
    if wrap:
        g[-1, :], g[:, -1] = g[0, :], g[:, 0]
    y = np.linspace(0, cells, h, endpoint=False)
    x = np.linspace(0, cells, w, endpoint=False)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
    b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def _mix(p, i, j, t):
    t = np.clip(t, 0, 1)[..., None]
    return p[i] * (1 - t) + p[j] * t


def pattern(sym, e):
    w, h = int(e["w"]), int(e["h"])
    rng = np.random.default_rng(h32("glidewire-tex", sym))
    p = _pal(e)
    pool = PATTERNS
    if str(e.get("yaml", "")).endswith("_textures") or e.get("yaml") == "course_track_gfx":
        pool = ["panel", "chevron", "hex", "ribs", "bands", "brushed", "grate", "terrain", "panel", "brushed"]
    kind = pool[int(rng.integers(len(pool)))]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    base, hi = int(rng.integers(0, 2)), int(rng.integers(1, 3))
    acc = int(rng.integers(3, 5))
    if kind == "panel":
        nx, ny = int(rng.choice([1, 2, 4])), int(rng.choice([1, 2, 4]))
        fx, fy = (u * nx) % 1, (v * ny) % 1
        edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
        img = _mix(p, base, hi, 0.35 + 0.3 * (fy < 0.5))
        img = np.where((edge < 0.04)[..., None], p[0], img)
        img = np.where(((edge > 0.04) & (edge < 0.08))[..., None], p[2] * 0.8 + img * 0.2, img)
        rivet = ((fx - 0.12) ** 2 + (fy - 0.12) ** 2 < 0.003)
        img = np.where(rivet[..., None], p[2], img)
    elif kind == "chevron":
        k = rng.choice([2, 3, 4])
        t = ((u * k + np.abs(v - 0.5) * 2) % 1) < 0.5
        img = np.where(t[..., None], p[acc], p[base])
    elif kind == "hex":
        s = rng.choice([4.0, 6.0, 8.0])
        q = u * s * 2
        r = v * s * 2 / np.sqrt(3) * (h / max(w, 1)) * (w / max(h, 1))
        cell = np.abs(((q + (np.floor(r) % 2) * 0.5) % 1) - 0.5) + np.abs((r % 1) - 0.5)
        img = _mix(p, base, hi, cell)
        img = np.where((cell > 0.85)[..., None], p[acc], img)
    elif kind == "ribs":
        k = rng.choice([4, 6, 8])
        t = np.abs(np.sin(np.pi * u * k))
        img = _mix(p, 0, hi, t ** 0.5)
    elif kind == "rings":
        cx, cy = 0.5, 0.5
        d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2) * rng.choice([4, 6, 8])
        img = _mix(p, base, acc, (np.sin(2 * np.pi * d) + 1) / 2)
    elif kind == "dots":
        k = rng.choice([4, 8])
        fx, fy = (u * k) % 1 - 0.5, (v * k) % 1 - 0.5
        img = np.where((fx * fx + fy * fy < 0.08)[..., None], p[acc], p[base])
    elif kind == "bands":
        k = rng.choice([2, 3, 5])
        t = np.floor(v * k) / max(k - 1, 1)
        img = _mix(p, 0, 2, t)
        img = np.where(((v * k) % 1 < 0.08)[..., None], p[acc], img)
    elif kind == "brushed":
        n = _noise(w, h, rng, 8)
        streak = _noise(w, h, rng, 32)
        img = _mix(p, 1, 2, 0.4 * n + 0.2 * streak)
    elif kind == "terrain":
        n = 0.6 * _noise(w, h, rng, 4) + 0.4 * _noise(w, h, rng, 8)
        img = _mix(p, 0, 1, n * 1.2)
        img = np.where((n > 0.7)[..., None], p[2] * 0.6 + img * 0.4, img)
    else:   # grate
        k = rng.choice([4, 8])
        t = (((u * k) % 1) < 0.2) | (((v * k) % 1) < 0.2)
        img = np.where(t[..., None], p[1], p[0])
    return np.clip(img, 0, 255)


def _shape_kind(sym, rng):
    s = sym.lower()
    try:
        from .fonts import char_of
        if char_of(sym)[1]:
            return "rect"          # glyph cells: the font painter fits our glyph into the cell
    except Exception:
        pass
    for hint, k in SHAPE_HINTS:
        if hint in s:
            return k
    return ["disc", "ring", "star", "puff", "diamond", "leaf", "blob"][int(rng.integers(7))]


def shape(sym, e):
    """Our coverage mask 0..1 for a transparent texture; aims at the slot's coarse cover fraction."""
    w, h = int(e["w"]), int(e["h"])
    rng = np.random.default_rng(h32("glidewire-shape", sym))
    kind = _shape_kind(sym, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + 0.5) / w * 2 - 1, (yy + 0.5) / h * 2 - 1
    r = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    if kind == "rect":
        m = (np.abs(u) < 0.94) & (np.abs(v) < 0.94)
    elif kind == "shield":
        m = (np.abs(u) < 0.9) & (v > -0.92) & ((v < 0.25) | (np.abs(u) < 0.9 * (0.95 - v) / 0.7))
    elif kind == "ring":
        m = np.abs(r - 0.7) < 0.2
    elif kind in ("star", "burst"):
        k = int(rng.integers(5, 9))
        m = r < (0.55 + 0.4 * np.abs(np.cos(ang * k / 2)) ** (3 if kind == "star" else 1))
    elif kind == "puff":
        m = np.zeros_like(r, bool)
        for _ in range(int(rng.integers(4, 7))):
            cx, cy, rr = rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.4), rng.uniform(0.3, 0.55)
            m |= (u - cx) ** 2 + (v - cy) ** 2 < rr * rr
    elif kind == "diamond":
        m = np.abs(u) + np.abs(v) < 0.95
    elif kind == "leaf":
        m = (np.abs(u) < (1 - (v + 1) / 2) ** 0.6 * 0.8) & (v > -0.95)
    elif kind == "wedge":
        m = (np.abs(v) < 0.55 * (u + 1) / 2 + 0.05) & (u > -0.95) & (u < 0.95)
    elif kind == "figure":
        head = u * u + (v + 0.62) ** 2 < 0.07
        body = (np.abs(u) < 0.25 + 0.45 * np.clip((v + 0.35) / 0.4, 0, 1)) & (v > -0.4)
        m = head | body
    elif kind == "streak":
        m = np.abs(v) < 0.35 * (1 - np.abs(u))
    elif kind == "blob":
        m = r < 0.75 + 0.15 * np.sin(ang * int(rng.integers(2, 5)) + rng.uniform(0, 6))
    else:
        m = r < 0.9
    a = m.astype(np.float32)
    cover = e.get("cover")
    if kind not in ("rect", "shield") and cover is not None and a.mean() > 0 and float(cover) < 0.995:
        # grow or shrink a little toward the slot's cover fraction (a structural fact of the slot)
        from scipy import ndimage
        for _ in range(3):
            if a.mean() < float(cover) - 0.08:
                a = ndimage.binary_dilation(a > 0.5).astype(np.float32)
            elif a.mean() > float(cover) + 0.08:
                a = ndimage.binary_erosion(a > 0.5).astype(np.float32)
    return a


def texture(sym, e):
    img = pattern(sym, e)
    a = np.full(img.shape[:2], 255.0, np.float32)
    if e.get("alpha"):
        a = shape(sym, e) * 255.0
    out = np.concatenate([img, a[..., None]], -1)
    if e["fmt"] in ("I4", "I8", "IA4", "IA8", "IA16"):
        lum = out[..., :3].mean(-1, keepdims=True)
        out[..., :3] = lum
    return np.clip(out, 0, 255).astype(np.uint8)


def grid(sym, e, n=4):
    img = texture(sym, e).astype(np.float32)
    h, w = img.shape[:2]
    out = []
    for j in range(n):
        for i in range(n):
            c = img[j * h // n:max(j * h // n + 1, (j + 1) * h // n), i * w // n:max(i * w // n + 1, (i + 1) * w // n)]
            out.append([int(x) for x in c.reshape(-1, 4).mean(0)])
    return out


def alpha2(sym, e):
    a = shape(sym, e) if e.get("alpha") else np.ones((int(e["h"]), int(e["w"])), np.float32)
    q = np.clip(np.round(a * 3), 0, 3).astype(np.uint8).reshape(-1)
    q = np.concatenate([q, np.zeros((-len(q)) % 4, np.uint8)])
    b = (q[0::4] << 6) | (q[1::4] << 4) | (q[2::4] << 2) | q[3::4]
    return bytes(b.astype(np.uint8)).hex()
