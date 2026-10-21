"""HUD and menu widgets with meaning, drawn by us: race position digits and suffixes, place markers, lap and
racer-count digit strips, the LED countdown, machine stats and weights, small tags.

Text is fitted into the box of the kept alpha outline (so the game's placement still lines up) and coloured
from the kept grid (vertical gradient of the brightest cells), with a dark rim for legibility.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter

from cleanroom.decomp.gen import unpack_alpha2

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
SS = 8

FACE = {"heavy": ("RussoOne-Regular.ttf", None), "italic": ("Exo2-Italic[wght].ttf", 900),
        "bold": ("Rubik.ttf", 800), "pixel": ("PressStart2P-Regular.ttf", None)}


def _font(kind, px):
    fn, w = FACE[kind]
    f = ImageFont.truetype(os.path.join(FONTS, fn), max(4, int(px)))
    if w:
        f.set_variation_by_axes([w])
    return f


def _grid(e):
    g = np.asarray(e["grid"], np.float32)
    n = int(round(len(g) ** 0.5))
    return g.reshape(n, n, 4)


def gradient(e, fallback=((255, 255, 255), (255, 255, 255))):
    """Top and bottom ink colours: the brightest opaque cells of the first and last opaque grid rows."""
    g = _grid(e)
    a = g[..., 3:4] / 255.0
    ok = a[..., 0] > 0.15
    if not ok.any():
        return np.array(fallback[0], np.float32), np.array(fallback[1], np.float32)
    rgb = np.clip(g[..., :3] / np.maximum(a, 1e-3), 0, 255)
    rows = [r for r in range(len(g)) if ok[r].any()]

    def best(r):
        c = rgb[r][ok[r]]
        c = c[np.argmax(c.sum(1))]
        return c * (230.0 / max(c.max(), 1.0)) if c.max() < 150 else c

    return best(rows[0]), best(rows[-1])


def bbox(e, thr=40):
    if "alpha2" not in e:
        return 0, e["w"], 0, e["h"]
    a = unpack_alpha2(e["alpha2"], e["w"], e["h"])
    ys, xs = np.nonzero(a > thr)
    if len(xs) == 0:
        return 0, e["w"], 0, e["h"]
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def text_mask(text, w, h, box, kind, squeeze=1.0):
    """Anti-aliased mask of `text` fitted into box (x0, x1, y0, y1) of a w x h texture."""
    x0, x1, y0, y1 = box
    W, H = w * SS, h * SS
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    bw, bh = (x1 - x0) * SS, (y1 - y0) * SS
    lo, hi, best = 4, H * 2, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = _font(kind, mid)
        b = d.textbbox((0, 0), text, font=f)
        if (b[2] - b[0]) * squeeze <= bw and b[3] - b[1] <= bh:
            best, lo = (f, b), mid + 1
        else:
            hi = mid - 1
    if best is None:
        f = _font(kind, 4)
        best = (f, d.textbbox((0, 0), text, font=f))
    f, b = best
    tw = b[2] - b[0]
    # draw, then stretch horizontally to fill the box width (condensed digits look like the HUD)
    tmp = Image.new("L", (int(tw) + 4 * SS, H), 0)
    ImageDraw.Draw(tmp).text((2 * SS - b[0], (y0 + y1) / 2 * SS - (b[1] + b[3]) / 2), text, font=f, fill=255)
    tmp = tmp.crop((2 * SS, 0, 2 * SS + int(tw), H))
    tw2 = int(min(bw, tw / squeeze if squeeze < 1 else tw))
    tmp = tmp.resize((max(1, tw2), H), Image.LANCZOS)
    img.paste(tmp, (int((x0 + x1) / 2 * SS - tw2 / 2), 0))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def colourise(m, top, bot, rim=(16, 16, 32), rim_px=1):
    h, w = m.shape
    t = np.linspace(0, 1, h)[:, None, None]
    rgb = np.broadcast_to(np.asarray(top)[None, None] * (1 - t) + np.asarray(bot)[None, None] * t, (h, w, 3)).copy()
    if rim is not None and rim_px:
        grown = maximum_filter(m, size=2 * rim_px + 1)
        edge = np.clip(grown - m, 0, 1)
        rgb = rgb * (1 - edge[..., None]) + np.asarray(rim, np.float32)[None, None] * edge[..., None]
        m = np.maximum(m, grown)
    return np.dstack([rgb, np.clip(m * 255 * 1.2, 0, 255)]).astype(np.uint8)


def fitted(e, text, kind="heavy", squeeze=1.0, rim=True):
    top, bot = gradient(e)
    m = text_mask(text, e["w"], e["h"], bbox(e), kind, squeeze)
    return colourise(m, top, bot, rim_px=1 if rim and e["h"] >= 10 else 0)


def strip(e, cells, kind="heavy"):
    """A vertical strip of equal cells, one glyph each (lap counter, racer digits)."""
    w, h = e["w"], e["h"]
    ch = h // len(cells)
    top, bot = gradient(e)
    out = np.zeros((h, w, 4), np.uint8)
    for i, c in enumerate(cells):
        m = text_mask(c, w, ch, (0, w, 1, ch - 1), kind)
        out[i * ch:(i + 1) * ch] = colourise(m, top, bot, rim_px=1)
    return out


LED = {"1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
       "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
       "3": [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
       "GO": ["#####.#####", "#.....#...#", "#.....#...#", "#.###.#...#", "#...#.#...#", "#...#.#...#", "#####.#####"]}


def led(e, key, on):
    """Dot-matrix countdown panel: lit dots on a dark framed panel."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([SS, SS, W - SS, H - SS], radius=3 * SS, fill=(18, 18, 22, 255), outline=(120, 120, 130, 255),
                        width=SS)
    pat = LED[key]
    rows, cols = len(pat), len(pat[0])
    cell = min((W - 8 * SS) / cols, (H - 8 * SS) / rows)
    ox, oy = (W - cell * cols) / 2, (H - cell * rows) / 2
    for r, row in enumerate(pat):
        for c, px in enumerate(row):
            cx, cy = ox + (c + 0.5) * cell, oy + (r + 0.5) * cell
            lit = px == "#"
            rr = cell * (0.46 if lit else 0.3)
            col = on if lit else (34, 28, 24)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=tuple(int(v) for v in col) + (255,))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


def marker(e, text, arrow=True):
    """Place marker: '1ST' over a down arrow (race view), colour from the grid."""
    w, h = e["w"], e["h"]
    top, bot = gradient(e)
    x0, x1, y0, y1 = bbox(e)
    th = (y1 - y0) * (0.55 if arrow else 1.0)
    m = text_mask(text, w, h, (x0, x1, y0, int(round(y0 + th))), "heavy")
    if arrow:
        W, H = w * SS, h * SS
        a = Image.new("L", (W, H), 0)
        cx = (x0 + x1) / 2 * SS
        ay0 = (y0 + th + 0.5) * SS
        aw = (x1 - x0) * SS * 0.32
        ImageDraw.Draw(a).polygon([(cx - aw, ay0), (cx + aw, ay0), (cx, y1 * SS)], fill=255)
        m = np.maximum(m, np.asarray(a.resize((w, h), Image.LANCZOS), np.float32) / 255.0)
    return colourise(m, top, top * 0.7 + bot * 0.3, rim_px=1)


def tag(e, text, bg=(245, 245, 245), fg=(20, 20, 20)):
    """Rounded pill with dark text (the 'Ghost' marker)."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=bg + (255,))
    f = _font("bold", H * 0.62)
    b = d.textbbox((0, 0), text, font=f)
    d.text(((W - (b[2] + b[0])) / 2, (H - (b[3] + b[1])) / 2), text, font=f, fill=fg + (255,))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


def panel_rows(e, rows):
    """BODY / BOOST / GRIP stat panel: labels with an underline arrow on a dark panel."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (22, 22, 26, 255))
    d = ImageDraw.Draw(img)
    rh = H / len(rows)
    f = _font("heavy", rh * 0.55)
    for i, t in enumerate(rows):
        y = i * rh
        d.line([(W * 0.04, y + rh * 0.15), (W * 0.04, y + rh * 0.85)], fill=(235, 235, 235, 255), width=SS)
        d.text((W * 0.08, y + rh * 0.12), t, font=f, fill=(245, 245, 245, 255))
        d.line([(W * 0.04, y + rh * 0.85), (W * 0.72, y + rh * 0.85)], fill=(235, 235, 235, 255), width=SS)
        d.polygon([(W * 0.66, y + rh * 0.72), (W * 0.74, y + rh * 0.85), (W * 0.66, y + rh * 0.88)],
                  fill=(235, 235, 235, 255))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


def energy_bar(e):
    """Energy meter frame: a bevelled tab labelled ENERGY over a dark slot the game fills."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tab = [(W * 0.16, H * 0.42), (W * 0.24, 0), (W * 0.76, 0), (W * 0.84, H * 0.42)]
    d.polygon(tab, fill=(196, 198, 206, 255), outline=(90, 92, 100, 255))
    d.rectangle([0, H * 0.38, W - 1, H - 1], fill=(150, 152, 160, 255), outline=(80, 82, 90, 255), width=SS)
    d.rectangle([W * 0.03, H * 0.55, W * 0.97, H * 0.9], fill=(10, 40, 24, 255), outline=(30, 30, 36, 255), width=SS)
    f = _font("heavy", H * 0.38)
    b = d.textbbox((0, 0), "ENERGY", font=f)
    d.text(((W - (b[0] + b[2])) / 2, H * 0.21 - (b[1] + b[3]) / 2), "ENERGY", font=f, fill=(40, 40, 50, 255))
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


def helmet_icon(e):
    """Our racing-helmet icon: teal shell, silver centre stripe, green wrap visor (Rook Valdane's colours)."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ink = (20, 10, 16, 255)
    d.ellipse([W * 0.08, H * 0.06, W * 0.92, H * 0.9], fill=(20, 130, 130, 255), outline=ink, width=SS)
    d.rectangle([W * 0.44, H * 0.08, W * 0.56, H * 0.5], fill=(225, 230, 236, 255))
    d.chord([W * 0.12, H * 0.42, W * 0.88, H * 0.98], 0, 180, fill=(60, 230, 120, 255), outline=ink, width=SS)
    d.ellipse([W * 0.06, H * 0.48, W * 0.2, H * 0.7], fill=(225, 230, 236, 255), outline=ink, width=SS)
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.uint8).copy()


SUFFIX = {"ND": "nd", "RD": "rd", "ST": "st", "TH": "th"}
PLACE = {"First": "1ST", "Second": "2ND", "Third": "3RD", "Fourth": "4TH", "Fifth": "5TH", "Sixth": "6TH"}


def hook(sym, e):
    if "w" not in e:
        return None
    if sym.startswith("aPosition") and sym[9:-3].isdigit():
        return fitted(e, sym[9:-3], "italic")
    if sym.startswith("aPositionOrdinalSuffix"):
        return fitted(e, SUFFIX[sym[22:24]], "italic")
    if sym.startswith("aFinalResultPositionSuffix"):
        return fitted(e, SUFFIX[sym[26:28]], "heavy")
    if sym.startswith("aFinalResultPosition") and sym[20:-3].isdigit():
        return fitted(e, sym[20:-3], "heavy")
    if sym.startswith("aMachineWeight") and sym[14:-3].isdigit():
        return fitted(e, sym[14:-3], "pixel", rim=False)
    if sym.startswith("aMachineStat") and len(sym) == 16:
        img = fitted(e, sym[12], "italic", rim=False)
        bg = np.zeros_like(img)
        bg[..., :3], bg[..., 3] = 30, 255
        a = img[..., 3:4] / 255.0
        bg[..., :3] = (img[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.uint8)
        return bg
    if sym.startswith("aPortraitPosition"):
        img = np.zeros((e["h"], e["w"], 4), np.uint8)
        img[..., :3], img[..., 3] = (30, 40, 150), 255
        t = fitted(dict(e, alpha2=None) if False else {k: v for k, v in e.items() if k != "alpha2"},
                   PLACE[sym[17:-3]], "pixel", rim=False)
        a = t[..., 3:4] / 255.0
        img[..., :3] = (255 * a + img[..., :3] * (1 - a)).astype(np.uint8)
        return img
    for k, t in (("aFirstPlaceMarker", "1ST"), ("aSecondPlaceMarker", "2ND"), ("aThirdPlaceMarker", "3RD")):
        if sym.startswith(k):
            return marker(e, t)
    if sym == "aRivalMarkerTex":
        return marker(e, "RIVAL")
    if sym == "aRacersLeftTex":
        return fitted(e, "LEFT", "heavy")
    if sym == "aHudTimeTex":
        return fitted(e, "TIME", "heavy")
    if sym == "aReverseTex":
        return fitted(e, "REVERSE", "heavy")
    if sym == "aBoostTex":
        return fitted(e, "BOOST", "pixel", rim=False)
    if sym == "aLapCounterSymbolsTex":
        return strip(e, ["/", "1", "2", "3", "4", "5"])
    if sym == "aTotalRacerDigitsTex":
        return strip(e, list("0123456789/"))
    if sym in ("aCountdown1Tex", "aCountdown2Tex", "aCountdown3Tex"):
        return led(e, sym[10], (255, 170, 30))
    if sym == "aCountdownGoTex":
        return led(e, "GO", (60, 255, 80))
    if sym == "aHasGhostMarkerTex":
        return tag(e, "Ghost")
    if sym in ("aMachineBodyBoostGripTex", "aMachineBodyBoostGripSmallTex"):
        return panel_rows(e, ["BODY", "BOOST", "GRIP"])
    if sym == "aStartStopGuideTex":
        return fitted(e, "(B) START  (A) STOP", "bold", rim=False)
    if sym == "aHudEnergyTex":
        return energy_bar(e)
    if sym == "aOptionsFalconHelmetTex":
        return helmet_icon(e)
    if sym == "aCheckMarker1PTex":
        return marker(e, "CHECK")
    return None
