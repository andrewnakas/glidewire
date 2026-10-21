"""Title-screen pictures, our own compositions: the crowd of pilots (aTitleBackgroundMainTex), the comic page
(aTitleBackgroundComicTex) and our Arc Lancer craft on black (aTitleBackgroundFalconTex).

Pilots come from pilots.py (our cast); background colours come from our kit (theme.py; no retail grid since
the 2026-10 rebuild). Everything else — lightning, panel layout, speech lines, the craft drawing — is ours.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import pilots

SS = 4
INK = (18, 14, 22, 255)


def grid_bg(e, blur=10):
    rgb, _ = pilots.grid_rgb(e)
    im = Image.fromarray(rgb.astype(np.uint8)).resize((e["w"], e["h"]), Image.BICUBIC)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)), np.float32)


def lightning(d, x0, y0, x1, y1, rng, width, col):
    pts = [(x0, y0)]
    n = 9
    for i in range(1, n):
        t = i / n
        pts.append((x0 + (x1 - x0) * t + rng.uniform(-1, 1) * abs(y1 - y0) * 0.12, y0 + (y1 - y0) * t))
    pts.append((x1, y1))
    d.line(pts, fill=(120, 170, 255, 255), width=width * 3, joint="curve")
    d.line(pts, fill=col, width=width, joint="curve")


def paste(canvas, img, x, y):
    """alpha-over an RGBA uint8 image onto a float RGB canvas at integer (x, y)"""
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = sub[..., 3:4] / 255.0
    canvas[y0:y1, x0:x1] = sub[..., :3] * a + canvas[y0:y1, x0:x1] * (1 - a)


CROWD_BACK = ["BioRex", "Beastman", "Octoman", "Draq", "TheSkull", "Leon", "Baba", "MightyGazelle", "Pico",
              "BlackShadow", "Zoda", "MrEad", "Billy", "GomarAndShioh"]
CROWD_MID = ["JodySummer", "DrStewart", "SamuraiGoroh", "JamesMcCloud", "KateAlen", "MichaelChain", "SilverNeelsen"]


def main_title(e):
    w, h = e["w"], e["h"]
    rng = np.random.default_rng(7)
    canvas = grid_bg(e, 14) * 0.8 + np.array([60, 20, 10], np.float32) * 0.2
    # lightning in the sky
    sky = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(sky)
    for x in (0.08, 0.3, 0.72, 0.93):
        lightning(d, x * w * SS, 0, (x + rng.uniform(-0.08, 0.08)) * w * SS, h * 0.45 * SS, rng, SS, (250, 250, 255, 255))
    paste(canvas, np.asarray(sky.resize((w, h), Image.LANCZOS)), 0, 0)
    # back row: small busts, darkened
    for i, name in enumerate(CROWD_BACK):
        c = pilots.brief(name)
        s = 46
        img = pilots.bust(s, s, c, s / 2, s * 0.45, s / 52).astype(np.float32)
        img[..., :3] *= 0.62
        x = int((i + 0.5) / len(CROWD_BACK) * w - s / 2 + rng.uniform(-4, 4))
        y = int(h * (0.12 if i % 2 else 0.2) + rng.uniform(-4, 4))
        paste(canvas, img.astype(np.uint8), x, y)
    # middle row
    for i, name in enumerate(CROWD_MID):
        c = pilots.brief(name)
        s = 72
        img = pilots.bust(s, s, c, s / 2, s * 0.45, s / 52).astype(np.float32)
        img[..., :3] *= 0.85
        x = int((i + 0.5) / len(CROWD_MID) * w - s / 2)
        if abs(i - len(CROWD_MID) // 2) <= 0:
            continue
        paste(canvas, img.astype(np.uint8), x, int(h * 0.36))
    # front: Rook Valdane (our lead), big, centre
    s = 150
    img = pilots.bust(s, s, pilots.brief("CaptainFalcon"), s / 2, s * 0.44, s / 50)
    paste(canvas, img, w // 2 - s // 2, int(h * 0.42))
    out = np.dstack([np.clip(canvas, 0, 255), np.full((h, w), 255.0)])
    return out.astype(np.uint8)


COMIC = [("CaptainFalcon", "SHOW ME YOUR MOVES!", (0, 0, 0.55, 0.5)),
         ("SamuraiGoroh", "I'LL TAKE YOU AT THE NEXT CURVE!", (0.55, 0, 1, 0.5)),
         ("Zoda", "EAT MY DUST!", (0, 0.5, 0.35, 1)),
         ("Pico", "YOUR TIME IS UP!", (0.35, 0.5, 0.7, 1)),
         ("BlackShadow", "HA HA HA!", (0.7, 0.5, 1, 1))]


def comic(e):
    w, h = e["w"], e["h"]
    canvas = np.zeros((h, w, 3), np.float32) + 12
    bgc = grid_bg(e, 6)
    for i, (name, line, (a, b, c2, d2)) in enumerate(COMIC):
        x0, y0, x1, y1 = int(a * w) + 3, int(b * h) + 3, int(c2 * w) - 3, int(d2 * h) - 3
        pw, ph = x1 - x0, y1 - y0
        tint = np.array([[230, 90, 60], [70, 120, 220], [120, 60, 170], [60, 170, 90], [200, 170, 50]][i], np.float32)
        panel = bgc[y0:y1, x0:x1] * 0.4 + tint * 0.6
        # speed lines
        yy, xx = np.mgrid[0:ph, 0:pw]
        ang = np.arctan2(yy - ph * 0.6, xx - pw * 0.5)
        panel *= (0.85 + 0.15 * (np.sin(ang * 22) > 0))[..., None]
        canvas[y0:y1, x0:x1] = panel
        s = int(min(pw, ph * 1.2))
        img = pilots.bust(s, s, pilots.brief(name), s / 2, s * 0.46, s / 50)
        paste(canvas, img, x0 + (pw - s) // 2 + (pw // 6 if i % 2 else -pw // 8), y0 + ph - int(s * 0.92))
        bub = pilots.bubble(int(pw * 1.3), int(ph * 0.8), line)
        paste(canvas, bub, x0 - int(pw * 0.02), y0 - int(ph * 0.05))
        canvas[y0:y1, x0:x0 + 2] = canvas[y0:y1, x1 - 2:x1] = 245
        canvas[y0:y0 + 2, x0:x1] = canvas[y1 - 2:y1, x0:x1] = 245
    return np.dstack([np.clip(canvas, 0, 255), np.full((h, w), 255.0)]).astype(np.uint8)


def falcon_machine(e):
    """Our Arc Lancer (teal hull, orange stripe, cyan canopy), three-quarter view from the front left, boosters glowing, on black with speed streaks."""
    w, h = e["w"], e["h"]
    W, H = w * SS, h * SS
    im = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(3)
    for _ in range(40):
        y = rng.uniform(0.15, 0.85) * H
        x = rng.uniform(0, 1) * W
        d.line([(x, y), (x + rng.uniform(0.1, 0.35) * W, y - 0.08 * W * 0.3)], fill=(40, 60, 120, 255),
               width=int(SS * rng.uniform(0.5, 1.5)))

    def P(pts):
        return [(x * W, y * H) for x, y in pts]

    blue, dark, white, red, yel = (20, 140, 150), (10, 70, 80), (235, 238, 245), (250, 140, 40), (120, 240, 250)
    ol = (8, 10, 30)
    # boosters (rear, upper right) with flames
    for bx, by in ((0.8, 0.36), (0.87, 0.47)):
        d.ellipse(P([(bx - 0.045, by - 0.06), (bx + 0.045, by + 0.06)]), fill=(90, 100, 120), outline=ol, width=SS)
        d.ellipse(P([(bx - 0.025, by - 0.035), (bx + 0.025, by + 0.035)]), fill=(120, 230, 255))
        d.polygon(P([(bx + 0.01, by - 0.03), (bx + 0.16, by - 0.07), (bx + 0.02, by + 0.03)]), fill=(150, 240, 255, 200))
    # side wing / fin
    d.polygon(P([(0.52, 0.34), (0.8, 0.14), (0.86, 0.18), (0.7, 0.4)]), fill=white, outline=ol, width=SS)
    d.polygon(P([(0.62, 0.3), (0.79, 0.18), (0.82, 0.2), (0.7, 0.33)]), fill=red)
    # main hull: long wedge from the nose (lower left) to the tail (upper right)
    hull = [(0.08, 0.72), (0.3, 0.55), (0.62, 0.36), (0.9, 0.32), (0.93, 0.5), (0.7, 0.62), (0.36, 0.76), (0.12, 0.8)]
    d.polygon(P(hull), fill=blue, outline=ol, width=2 * SS)
    d.polygon(P([(0.12, 0.8), (0.36, 0.76), (0.7, 0.62), (0.93, 0.5), (0.92, 0.58), (0.7, 0.7), (0.34, 0.84),
                 (0.1, 0.84)]), fill=dark, outline=ol, width=SS)
    # stripes
    d.polygon(P([(0.14, 0.73), (0.62, 0.45), (0.9, 0.4), (0.9, 0.43), (0.63, 0.49), (0.16, 0.76)]), fill=white)
    d.polygon(P([(0.18, 0.76), (0.63, 0.51), (0.9, 0.45), (0.9, 0.47), (0.64, 0.53), (0.2, 0.78)]), fill=red)
    # cockpit canopy
    d.polygon(P([(0.38, 0.52), (0.5, 0.4), (0.62, 0.38), (0.6, 0.47), (0.46, 0.55)]), fill=yel, outline=ol, width=SS)
    d.polygon(P([(0.44, 0.49), (0.51, 0.43), (0.56, 0.42), (0.52, 0.46)]), fill=(255, 250, 200))
    # nose light and number plate
    d.ellipse(P([(0.09, 0.73), (0.13, 0.77)]), fill=(255, 250, 220), outline=ol, width=SS)
    d.polygon(P([(0.66, 0.5), (0.76, 0.46), (0.77, 0.52), (0.67, 0.56)]), fill=white, outline=ol, width=SS)
    from .signs import fit_text
    fit_text(d, "01", "heavy", (0.67 * W, 0.47 * H, 0.76 * W, 0.55 * H), (20, 20, 60, 255))
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.uint8).copy()


def hook(sym, e):
    if sym == "aTitleBackgroundMainTex":
        return main_title(e)
    if sym == "aTitleBackgroundComicTex":
        return comic(e)
    if sym == "aTitleBackgroundFalconTex":
        return falcon_machine(e)
    return None
