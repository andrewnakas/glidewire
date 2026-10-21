"""Pilot pictures: every character face / portrait slot, drawn from our own written briefs.

Nothing here is copied or traced from retail pixels. Each pilot is a short description in our own
words (CHARACTERS below: head template, colours, hair / helmet / eyewear / mouth / props), painted
procedurally with PIL at 8x supersampling and downsampled, with dark cel outlines so eyes and mouths
stay readable at 32 px. The only retail facts used are the kept texture facts from the spec
(size, format, 4x4 / 16x16 colour grid, 2-bit alpha outline).

Drawn:
  aPortrait<Name>Tex 32x32, aSmallPortrait<Name>Tex 40x40   head-and-shoulders bust in a white frame
  aFullPortrait<Name>Tex 180x245   kept alpha silhouette, cel-shaded suit colours from the 16x16 grid,
                                   our head from the brief placed at the top of the silhouette
  aEnding<Name>Tex 168x99          blurred grid backdrop + large bust + thin white frame
  aCountdownMrZeroMouth{Closed,Open}Tex 32x32   the announcer on his monitor (kept alpha outline)
aFinishFaceTex / aGameoverFaceTex are plain colour ramps (no face) and aFullPortraitBackgroundTex is a
backdrop: those stay on the grid default (hook returns None).

    python -m games.fzerox.pilots <out.png>     # quick sheet of all busts (dev)
"""
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 8
INK = (18, 14, 22, 255)
WHITE = (250, 250, 250)


# ------------------------------------------------------------------ briefs (our own words)
# Head template "human" is a face oval (local units: face half-width 16, chin at +21) with options:
#   skin, hair (colour), style (short/slick/long/curly/wild/bald/flattop/hood/none), helmet (dict),
#   eyes (normal/mask/shades/glasses/goggles/visor/paint/slit/red), iris, brows (normal/angry/none),
#   mouth (grin/smile/closed/open/lips/fangs/snarl), lips, beard, headset, headband, suit, collar, bg
# Other templates: monkey, lizard, rex, robot, ead, octo, alien, skull, draq, twin, dog, zero.
from .cast import CHARACTERS  # noqa: E402  (our own designs; 2026-10 rebuild)
ALIASES = {"CaptainFalconMaster": "CaptainFalcon", "CaptainFalconStandardExpert": "CaptainFalcon"}


# ------------------------------------------------------------------ painter
def rgba(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), a)


def sh(c, k):
    return tuple(int(max(0, min(255, v * k + (0 if k <= 1 else (k - 1) * 40)))) for v in c[:3])


def chaikin(pts, it=3, closed=True):
    pts = [tuple(p) for p in pts]
    for _ in range(it):
        out = []
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        if not closed:
            out.append(pts[0])
        for i in rng:
            a, b = pts[i], pts[(i + 1) % n]
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        if not closed:
            out.append(pts[-1])
        pts = out
    return pts


class Pen:
    """Draws in local head units: out_px = (ox + fx*k*x, oy + k*y); rendered at SS, outlines in output px."""

    def __init__(self, w, h, ol=0.85, bg=None):
        self.W, self.H = w, h
        self.im = Image.new("RGBA", (w * SS, h * SS), rgba(bg) if bg else (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.ow = max(1, int(round(ol * SS)))
        self.at(0, 0, 1)

    def at(self, ox, oy, k, flip=False):
        self.ox, self.oy, self.k, self.fx = ox, oy, k, (-1 if flip else 1)

    def P(self, x, y):
        return ((self.ox + self.fx * self.k * x) * SS, (self.oy + self.k * y) * SS)

    def box(self, x, y, rx, ry):
        (a, b), (c, d) = self.P(x - rx, y - ry), self.P(x + rx, y + ry)
        return [min(a, c), min(b, d), max(a, c), max(b, d)]

    def ell(self, x, y, rx, ry, c, ol=True):
        self.d.ellipse(self.box(x, y, rx, ry), fill=rgba(c) if c else None,
                       outline=INK if ol else None, width=self.ow if ol else 0)

    def ring(self, x, y, rx, ry, c, w):
        self.d.ellipse(self.box(x, y, rx, ry), outline=rgba(c), width=max(1, int(w * self.k * SS)))

    def poly(self, pts, c, ol=True, smooth=0):
        if smooth:
            pts = chaikin(pts, smooth)
        self.d.polygon([self.P(*p) for p in pts], fill=rgba(c) if c else None,
                       outline=INK if ol else None, width=self.ow if ol else 0)

    def line(self, pts, c=None, w=1.0, smooth=0, px=None):
        if smooth:
            pts = chaikin(pts, smooth, closed=False)
        wid = int(round(px * SS)) if px else int(round(w * self.k * SS))
        self.d.line([self.P(*p) for p in pts], fill=rgba(c) if c else INK, width=max(1, wid), joint="curve")

    def blob(self, x, y, rx, ry, c, ol=True, light=True):
        """shaded ellipse: dark rim bottom-right, lit body, highlight top-left"""
        self.ell(x, y, rx, ry, sh(c, 0.68), ol)
        self.ell(x - rx * 0.1, y - ry * 0.1, rx * 0.86, ry * 0.86, c, False)
        if light:
            self.ell(x - rx * 0.38, y - ry * 0.42, rx * 0.28, ry * 0.2, sh(c, 1.25), False)

    def result(self):
        return np.asarray(self.im.resize((self.W, self.H), Image.LANCZOS), np.uint8)


def sym(pts):
    """mirror a half outline given left-to-right on the left side (x<=0) into a closed symmetric shape"""
    return list(pts) + [(-x, y) for x, y in reversed(pts)]


# ------------------------------------------------------------------ parts
def face_pts(jaw=1.0, top=-20):
    j = jaw
    pts = []
    for i in range(0, 11):
        t = math.pi + math.pi * i / 10.0
        pts.append((16 * math.cos(t), -top * math.sin(t)))
    pts += [(16, 2), (15.5 * j, 9), (11 * j, 16), (4.5 * j, 20.5), (-4.5 * j, 20.5), (-11 * j, 16), (-15.5 * j, 9),
            (-16, 2)]
    return pts


def draw_face(p, c):
    skin = c["skin"]
    pts = face_pts(c.get("jaw", 1.0))
    p.poly(pts, sh(skin, 0.72), smooth=2)
    inner = [(x * 0.9 - 1.2, y * 0.93 - 0.8) for x, y in pts]
    p.poly(inner, skin, ol=False, smooth=2)
    p.ell(-7, -8, 5, 3, sh(skin, 1.12), ol=False)


def draw_ears(p, c):
    s = c["skin"]
    for x in (-16, 16):
        p.ell(x, 2, 3.2, 5, sh(s, 0.85))


def eye(p, x, y, iris=(60, 60, 60), c=None, narrow=False, lash=False, mirror=False):
    ry = 1.6 if narrow else 2.4
    p.ell(x, y, 3.8, ry, WHITE)
    p.ell(x + (0.4 if mirror else -0.4), y + 0.2, 1.8, min(ry, 2.1), iris, ol=False)
    p.ell(x + (0.4 if mirror else -0.4), y + 0.2, 0.9, min(ry * 0.7, 1.2), (10, 10, 14), ol=False)
    p.ell(x - 0.9, y - 0.8, 0.6, 0.5, (255, 255, 255), ol=False)
    s = 1 if x > 0 else -1
    p.line([(x - 4.2, y - ry * 0.6), (x, y - ry - 0.3), (x + 4.2, y - ry * 0.6)], w=1.3 if lash else 1.0, smooth=1)
    if lash:
        p.line([(x + s * 3.8, y - ry * 0.5), (x + s * 5.4, y - ry - 0.8)], w=0.9)


def brows(p, c, kind):
    col = sh(c.get("hair", (40, 30, 30)), 0.7) if kind != "white" else (240, 240, 244)
    if kind == "angry":
        for s in (-1, 1):
            p.poly([(s * 2, -3.2), (s * 11, -7.5), (s * 11, -5.2), (s * 2.5, -1.6)], col)
    elif kind == "white":
        for s in (-1, 1):
            p.poly([(s * 2, -4), (s * 6, -7.5), (s * 12, -6.5), (s * 11, -3.5), (s * 5, -4.2)], col, smooth=1)
    elif kind != "none":
        for s in (-1, 1):
            p.line([(s * 3, -5), (s * 7, -6.5), (s * 11, -5.5)], sh(c.get("hair", (40, 30, 30)), 0.6), w=1.4, smooth=1)


def nose(p, c, big=False):
    s = c["skin"]
    if big:
        p.ell(0, 5, 3.4, 3, sh(s, 0.95))
        p.ell(-0.8, 4, 1.2, 1, sh(s, 1.15), ol=False)
        return
    p.poly([(0.4, -1), (2.4, 5.5), (-1.2, 6.2)], sh(s, 0.75), ol=False)
    p.line([(2.4, 5.5), (0, 6.6), (-1.6, 6)], w=0.8)


def mouth(p, c, kind, opened=False):
    y = 12
    lips = c.get("lips")
    if opened and kind in ("closed", "smile", "lips", "none"):
        kind = "open"
    if kind == "grin":
        p.poly([(-8, y - 2), (8, y - 2), (5.5, y + 3), (-5.5, y + 3)], WHITE, smooth=1)
        p.line([(-7, y + 0.5), (7, y + 0.5)], w=0.6)
        if lips:
            p.line([(-8.5, y - 2.5), (8.5, y - 2.5)], lips, w=1.2)
    elif kind == "evil":
        p.poly([(-11, y - 5), (0, y - 1), (11, y - 5), (6, y + 5), (-6, y + 5)], (70, 10, 30), smooth=1)
        p.poly([(-9, y - 3.8), (9, y - 3.8), (7, y - 1.6), (-7, y - 1.6)], WHITE, ol=False)
        p.poly([(-6, y + 3.6), (6, y + 3.6), (5, y + 2), (-5, y + 2)], WHITE, ol=False)
        if lips:
            p.line([(-11, y - 5), (-6, y + 5.2), (6, y + 5.2), (11, y - 5)], lips, w=1.0)
    elif kind == "snarl":
        p.poly([(-7, y - 1.5), (7, y - 2), (6, y + 2.5), (-6, y + 2.5)], WHITE)
        for x in (-3.5, 0, 3.5):
            p.line([(x, y - 1.8), (x, y + 2.4)], w=0.6)
        p.line([(-7, y + 0.4), (7, y + 0.2)], w=0.6)
    elif kind == "open":
        p.poly([(-6, y - 2), (6, y - 2), (3.5, y + 5), (-3.5, y + 5)], (110, 20, 30), smooth=1)
        p.poly([(-5, y - 1.4), (5, y - 1.4), (4.4, y), (-4.4, y)], WHITE, ol=False)
        p.ell(0, y + 3.2, 2.6, 1.2, (220, 90, 100), ol=False)
        if lips:
            p.line([(-6.5, y - 2), (0, y - 2.6), (6.5, y - 2), (3.5, y + 5.4), (-3.5, y + 5.4), (-6.5, y - 2)], lips,
                   w=1.2)
    elif kind == "smile":
        p.line([(-6, y - 1.5), (-3, y + 0.8), (3, y + 0.8), (6, y - 1.5)], w=1.1, smooth=2)
    elif kind == "lips":
        p.poly([(-5.5, y - 0.3), (-2, y - 1.8), (0, y - 1.2), (2, y - 1.8), (5.5, y - 0.3), (2, y + 2), (-2, y + 2)],
               lips or (200, 60, 70), smooth=1)
        p.line([(-5, y - 0.2), (5, y - 0.2)], w=0.6)
    elif kind == "closed":
        p.line([(-5.5, y - 0.6), (0, y), (5.5, y - 0.6)], w=1.2, smooth=1)
        p.line([(-2.5, y + 2.6), (2.5, y + 2.6)], sh(c["skin"], 0.7), w=1.0)


def eyes(p, c, kind, iris):
    lash = c.get("lashes", False)
    if kind in ("normal", "narrow"):
        for x in (-7, 7):
            eye(p, x, -1, iris, narrow=kind == "narrow", lash=lash, mirror=x > 0)
    elif kind == "shades":
        for s in (-1, 1):
            p.poly([(s * 1.5, -3.5), (s * 13, -4), (s * 12.5, 0.5), (s * 8, 2), (s * 3, 1)], (20, 20, 26), smooth=1)
            p.line([(s * 10, -2.6), (s * 7, -2.6)], (130, 140, 170), w=0.9)
        p.line([(-2, -3), (2, -3)], (20, 20, 26), w=1.2)
    elif kind == "visor":
        p.poly([(-15, -4), (15, -4), (13.5, 1.5), (-13.5, 1.5)], iris, smooth=1)
        p.line([(-11, -2.4), (-4, -2.4)], sh(iris, 1.4), w=0.9)
    elif kind == "glasses":
        for x in (-7, 7):
            eye(p, x, -1, (70, 70, 90), narrow=True, mirror=x > 0)
            p.ring(x, -1, 5, 4.2, INK, 1.1)
            p.ring(x, -1, 4.2, 3.4, (220, 220, 230), 0.6)
        p.line([(-2, -1.5), (2, -1.5)], w=1)
    elif kind == "goggles":
        for x in (-7, 7):
            p.ell(x, -1.5, 5, 4.5, (80, 80, 90))
            p.ell(x, -1.5, 3.6, 3.2, iris)
            p.ell(x - 1.2, -2.8, 1.2, 1, (255, 255, 230), ol=False)
    elif kind == "paint":
        for s in (-1, 1):
            p.poly([(s * 1.5, -6), (s * 14, -7), (s * 13, 3), (s * 3, 2.5)], (22, 18, 20), ol=False, smooth=1)
            p.ell(s * 7, -1.5, 3.4, 1.8, WHITE)
            p.ell(s * 7, -1.3, 1.1, 1.3, (20, 20, 20), ol=False)
    elif kind == "red":
        for s in (-1, 1):
            p.poly([(s * 3, -1.5), (s * 11, -4), (s * 10, 0.5), (s * 4, 0.8)], (30, 10, 16))
            p.poly([(s * 4.5, -1), (s * 9.5, -2.8), (s * 9, -0.3), (s * 5, 0)], iris if c.get("iris") else (250, 40, 40),
                   ol=False)
    elif kind == "mask":
        pass  # the helmet visor carries the eyes


def helmet_back(p, c, h):
    col = h["col"]
    p.blob(0, -3, 21, 24, col)
    if h.get("horns"):
        for s in (-1, 1):
            p.poly([(s * 12, -18), (s * 26, -30), (s * 24, -24), (s * 19, -12)], (90, 86, 104))
    if h.get("cups"):
        for s in (-1, 1):
            p.blob(s * 20, 3, 5, 7, h["cups"])


def helmet_front(p, c, h):
    col = h["col"]
    # shell over the brow and down the cheeks, leaving the lower face open
    pts = [(-21, 10), (-22, -6), (-18, -20), (-8, -27), (8, -27), (18, -20), (22, -6), (21, 10), (15.5, 10),
           (14.5, -4), (8, -7), (0, -7.5), (-8, -7), (-14.5, -4), (-15.5, 10)]
    p.poly(pts, sh(col, 0.75), smooth=1)
    p.poly([(x * 0.9 - 1, y * 0.92 - 1) for x, y in pts[1:8]] + [(10, -9), (-12, -9)], col, ol=False, smooth=1)
    p.ell(-8, -19, 5, 3, sh(col, 1.3), ol=False)
    e, ec = h.get("emblem"), h.get("ecol", WHITE)
    if e == "falcon":
        p.poly([(0, -9), (-3, -13), (-11, -17), (-5, -18.5), (0, -16), (5, -18.5), (11, -17), (3, -13)], ec)
        p.poly([(0, -14.5), (-1.8, -19), (0, -23), (1.8, -19)], ec)
    elif e == "falcon_big":
        p.poly([(0, -8), (-5, -12), (-19, -14), (-13, -20), (-4, -18), (0, -26), (4, -18), (13, -20), (19, -14),
                (5, -12)], ec)
    elif e == "bolt":
        p.poly([(-5, -22), (5, -22), (-1, -15), (5, -15), (-5, -9), (-1, -14), (-6, -14)], ec)
    elif e == "A":
        p.ell(0, -16, 6.5, 6, ec)
        p.poly([(0, -21), (4, -12), (2, -12), (1, -14.5), (-1, -14.5), (-2, -12), (-4, -12)], col)
    elif e == "stripes":
        for s in (-1, 1):
            p.poly([(s * 3, -26), (s * 5, -26), (s * 12, -9), (s * 10, -8.5)], ec, ol=False)
    if h.get("wings"):
        for s in (-1, 1):
            p.poly([(s * 15, 4), (s * 17, -14), (s * 24, -30), (s * 24, -19), (s * 29, -24), (s * 27, -12),
                    (s * 31, -14), (s * 26, -2), (s * 21, 6)], (246, 246, 250))
            p.line([(s * 19, -12), (s * 23, -24)], (160, 160, 190), w=0.8)
            p.line([(s * 21, -6), (s * 27, -13)], (160, 160, 190), w=0.8)
    if h.get("visor") == "mask":
        p.poly([(-16, -8), (16, -8), (15.5, 1), (7, 3), (2.5, 1), (0, 3.5), (-2.5, 1), (-7, 3), (-15.5, 1)],
               (22, 20, 30))
        for s in (-1, 1):
            p.poly([(s * 3, -3), (s * 11.5, -5.5), (s * 11, -2.5), (s * 4, -0.8)], (246, 246, 250), ol=False)


def hair_back(p, c):
    st, hc = c.get("style"), c.get("hair")
    if st == "long":
        p.poly([(-22, 34), (-24, 0), (-19, -18), (0, -25), (19, -18), (24, 0), (22, 34)], sh(hc, 0.85), smooth=2)
    elif st == "curly":
        for (x, y, r) in [(-20, 22, 8), (20, 22, 8), (-24, 8, 9), (24, 8, 9), (-22, -8, 9), (22, -8, 9),
                          (-14, -20, 10), (14, -20, 10), (0, -24, 11)]:
            p.blob(x, y, r, r, hc)
    elif st == "hood":
        p.poly([(-30, 40), (-25, 0), (-20, -20), (0, -28), (20, -20), (25, 0), (30, 40)], sh(hc, 0.8), smooth=2)
    elif st == "wild":
        spikes = []
        for i in range(15):
            a = math.pi * (0.9 + 1.2 * i / 14.0)
            r = 29 if i % 2 == 0 else 20
            spikes.append((r * math.cos(a) * 0.95, -4 + r * math.sin(a) * 0.95))
        p.poly([(-17, 10)] + spikes + [(17, 10)], sh(hc, 0.85))


def hair_front(p, c):
    st, hc = c.get("style"), c.get("hair")
    if st in ("short", "slick"):
        top = -27 if st == "slick" else -25
        pts = [(-17, 2), (-18.5, -10), (-14, -21), (-4, top), (8, top + 1), (16, -20), (18.5, -10), (17, 2), (15, -6),
               (10, -11)]
        pts += ([(4, -9), (-1, -13), (-8, -12), (-14, -7)] if st == "slick" else
                [(5, -11), (0, -9.5), (-5, -11), (-10, -10), (-14, -6)])
        p.poly(pts, hc, smooth=2)
        p.line([(-10, -18), (0, -22), (8, -21)], sh(hc, 1.35), w=1.2, smooth=1)
    elif st == "long":
        p.poly([(-17, 12), (-19, -8), (-12, -21), (0, -24), (12, -21), (19, -8), (17, 12), (14.5, -2), (8, -12),
                (0, -8), (-9, -12), (-14.5, -2)], hc, smooth=2)
        p.line([(-11, -18), (-2, -21)], sh(hc, 1.35), w=1.2)
        if c.get("streak"):
            p.poly([(4, -22), (10, -21), (15, -4), (13, 10), (11, -6)], c["streak"], ol=False, smooth=1)
    elif st == "curly":
        for (x, y, r) in [(-12, -13, 7), (-4, -15, 7), (4, -15, 7), (12, -13, 7), (-16, -4, 5), (16, -4, 5)]:
            p.blob(x, y, r, r * 0.8, hc)
    elif st == "wild":
        p.poly([(-17, 2), (-18, -12), (-10, -20), (0, -21), (10, -20), (18, -12), (17, 2), (15, -5), (11, -3),
                (8, -9), (4, -5), (0, -11), (-4, -5), (-8, -9), (-11, -3), (-15, -5)], hc)
    elif st == "flattop":
        p.poly([(-17, -2), (-18, -14), (-17, -28), (17, -28), (18, -14), (17, -2), (15, -9), (-15, -9)], hc)
        p.line([(-14, -25), (12, -25)], sh(hc, 1.3), w=1.0)
        if c.get("stripe"):
            p.poly([(4, -28), (9, -28), (8, -14), (5, -14)], c["stripe"], ol=False)
    elif st == "hood":
        p.poly([(-21, 18), (-20, -8), (-15, -21), (0, -26), (15, -21), (20, -8), (21, 18), (16.5, 18), (16, -5),
                (10, -12), (0, -14), (-10, -12), (-16, -5), (-16.5, 18)], hc, smooth=1)
        p.line([(-12, -20), (0, -23)], sh(hc, 1.3), w=1.2)
    elif st == "wrap":
        p.poly([(-17, -6), (-19, -16), (-10, -28), (4, -30), (16, -24), (19, -12), (17, -6), (0, -9)], hc, smooth=2)
        sc = c.get("stripe", INK[:3])
        for x0 in (-12, -4, 4, 12):
            p.line([(x0 - 3, -10), (x0 + 1, -18), (x0 - 1, -26)], sc, w=1.6, smooth=1)
    elif st == "bald":
        p.ell(-6, -14, 5, 3, sh(c["skin"], 1.25), ol=False)
        if c.get("stripe"):
            p.poly([(-2, -20), (2, -20), (2.5, -10), (-2.5, -10)], c["stripe"], ol=False)
    elif st == "cap":
        p.poly([(-18, -4), (-19, -14), (-12, -24), (0, -27), (12, -24), (19, -14), (18, -4), (0, -7)], hc, smooth=1)
        p.poly([(-20, -6), (20, -6), (18, -2), (-18, -2)], sh(hc, 0.7))
        gc = c.get("goggles_up", (200, 200, 220))
        for x in (-7, 7):
            p.ell(x, -15, 5.5, 4.5, (80, 80, 100))
            p.ell(x, -15, 4, 3.2, gc)
            p.ell(x - 1.4, -16.2, 1.2, 0.9, (255, 255, 255), ol=False)
        p.line([(-2, -15), (2, -15)], w=1.2)
    if c.get("headband") == "sun":
        p.poly([(-18, -7), (18, -7), (17.5, -12.5), (-17.5, -12.5)], (246, 246, 246))
        p.ell(0, -9.8, 3, 2.6, (220, 30, 40), ol=False)
        for i in range(-3, 4):
            if i:
                p.poly([(0, -9.8), (i * 5.0 - 1.1, -12.5), (i * 5.0 + 1.1, -12.5)], (220, 30, 40), ol=False)
                p.poly([(0, -9.8), (i * 5.0 - 1.1, -7), (i * 5.0 + 1.1, -7)], (220, 30, 40), ol=False)


def beard(p, c):
    bc = c["beard"]
    p.poly([(-17, 0), (-19, 14), (-12, 26), (0, 31), (12, 26), (19, 14), (17, 0), (12, 6), (5, 7), (0, 5.5),
            (-5, 7), (-12, 6)], bc, smooth=2)
    p.poly([(-11, 11), (-5, 6.5), (0, 8), (5, 6.5), (11, 11), (6, 11.5), (0, 10.5), (-6, 11.5)], sh(bc, 0.95),
           smooth=1)
    p.line([(-3.5, 14.5), (3.5, 14.5)], (120, 40, 50), w=1.4)
    for x in (-8, 0, 8):
        p.line([(x, 17), (x * 0.9, 24)], sh(bc, 0.8), w=0.8)


def headset(p, c):
    col = c["headset"]
    for s in (-1, 1):
        p.blob(s * 17.5, 1, 3.8, 5.5, col)
    p.line([(-17, 5), (-12, 14), (-5, 15)], (60, 60, 70), w=1.0, smooth=1)
    p.ell(-5, 15, 1.6, 1.3, (60, 60, 70), ol=False)


def human(p, c, opened=False):
    h = c.get("helmet")
    hair_back(p, c)
    if h:
        helmet_back(p, c, h)
    elif c.get("style") not in ("hood", "wrap"):
        draw_ears(p, c)
    draw_face(p, c)
    if c.get("stubble"):
        p.poly([(-15, 7), (-10, 16), (-4, 20), (4, 20), (10, 16), (15, 7), (8, 13), (-8, 13)], sh(c["skin"], 0.82),
               ol=False, smooth=1)
    eyes(p, c, c.get("eyes", "normal"), c.get("iris", (60, 60, 70)))
    if not h:
        brows(p, c, c.get("brows", "normal"))
    nose(p, c, big=c.get("old", False) and c.get("beard") is not None)
    if c.get("old"):
        for s in (-1, 1):
            p.line([(s * 6, 5), (s * 9, 10)], sh(c["skin"], 0.65), w=0.7)
    if c.get("beard"):
        beard(p, c)
    else:
        mouth(p, c, c.get("mouth", "closed"), opened)
    if h:
        helmet_front(p, c, h)
    else:
        hair_front(p, c)
    if c.get("headset"):
        headset(p, c)


# ---- creature templates
def monkey(p, c, opened=False):
    f, fc = c["fur"], c["face"]
    for s in (-1, 1):
        p.blob(s * 20, 0, 6, 7, f)
        p.ell(s * 20, 0.5, 3.4, 4.2, fc, ol=False)
    p.blob(0, -2, 20, 22, f)
    p.poly([(-15, -3), (-8, -9), (0, -6), (8, -9), (15, -3), (16, 10), (8, 20), (-8, 20), (-16, 10)], fc, smooth=2)
    p.poly([(-4, -18), (-10, -30), (-2, -24), (0, -33), (3, -24), (11, -29), (6, -17)], c["tuft"], smooth=1)
    for x in (-6.5, 6.5):
        eye(p, x, -3, (60, 40, 20), narrow=True, mirror=x > 0)
    p.poly([(-11, -8), (-2, -6), (-2, -4.6), (-11, -6.2)], sh(f, 0.6), ol=False)
    p.poly([(11, -8), (2, -6), (2, -4.6), (11, -6.2)], sh(f, 0.6), ol=False)
    p.ell(-1.6, 3, 1, 0.8, INK[:3], ol=False)
    p.ell(1.6, 3, 1, 0.8, INK[:3], ol=False)
    p.poly([(-11, 6), (11, 6), (7, 18), (-7, 18)], (230, 110, 130), smooth=1)
    p.poly([(-9.5, 6.8), (9.5, 6.8), (8.5, 9.4), (-8.5, 9.4)], WHITE, ol=False)
    p.poly([(-6.5, 15.5), (6.5, 15.5), (5.5, 17.4), (-5.5, 17.4)], WHITE, ol=False)
    p.ell(0, 13.5, 3.5, 1.4, (170, 50, 70), ol=False)


def lizard(p, c, opened=False):
    f, sk = c["fur"], c["skin"]
    draw_face(p, dict(skin=sk, jaw=1.1))
    for s in (-1, 1):
        p.poly([(s * 3, -2), (s * 11, -4.5), (s * 10.5, -0.5), (s * 4, 0.5)], (240, 240, 230), ol=False)
    mouth(p, dict(skin=sk), "grin")
    # reptile head worn over the brow: snout forward, row of teeth along the lower rim
    p.poly([(-22, 6), (-22, -12), (-14, -26), (0, -31), (14, -26), (22, -12), (22, 6), (17, 4), (15, -6), (0, -8),
            (-15, -6), (-17, 4)], sh(f, 0.85), smooth=1)
    p.poly([(-12, -24), (0, -29), (12, -24), (8, -14), (-8, -14)], f, ol=False, smooth=1)
    for i in range(7):
        x = -12 + i * 4
        p.poly([(x - 1.6, -7.8), (x + 1.6, -7.8), (x, -3.8)], WHITE)
    for s in (-1, 1):
        p.ell(s * 11, -18, 3.4, 2.4, (250, 220, 40))
        p.ell(s * 11, -18, 0.8, 2, INK[:3], ol=False)
    p.line([(-6, -26), (6, -26)], c["light"], w=1.2)


def rex(p, c, opened=False):
    f, st = c["fur"], c["stripe"]
    # facing left: skull at right, jaws open towards the left
    p.poly([(26, 30), (16, -8), (10, -18), (-6, -20), (-24, -16), (-30, -11), (-28, -6), (-10, -4), (2, 0),
            (-18, 6), (-26, 8), (-24, 13), (-4, 14), (12, 22), (14, 30)], sh(f, 0.8), smooth=1)
    p.poly([(14, -6), (8, -15), (-6, -17), (-24, -13), (-27, -9), (-9, -7), (4, -3)], f, ol=False, smooth=1)
    p.poly([(-26, -6), (-10, -3), (2, 0), (-18, 6), (-26, 8)], (140, 30, 40), ol=False)
    for i in range(6):
        x = -25 + i * 4.5
        p.poly([(x, -6.5 + i * 0.5), (x + 2.4, -6.3 + i * 0.5), (x + 1.2, -3 + i * 0.5)], WHITE)
        p.poly([(x + 1, 7.5 - i * 0.9), (x + 3.4, 7.2 - i * 0.9), (x + 2.2, 4.2 - i * 0.9)], WHITE)
    for x in (0, 7, 14):
        p.line([(x, -16), (x + 3, -8)], st, w=1.5)
    p.ell(4, -11, 3.6, 2.6, (250, 214, 40))
    p.ell(4, -11, 0.8, 2.1, INK[:3], ol=False)
    p.line([(-1, -15), (8, -14)], w=1.3)
    p.ell(-26, -12, 1, 0.8, INK[:3], ol=False)


def robot(p, c, opened=False):
    f = c["fur"]
    for s in (-1, 1):
        p.blob(s * 19, 2, 5, 9, sh(f, 0.8))
    p.blob(0, -4, 19, 24, f)
    p.poly([(-13, -6), (13, -6), (11, 14), (5, 20), (-5, 20), (-11, 14)], c["light"])
    p.poly([(-15, -9), (15, -9), (13, -2), (-13, -2)], (24, 24, 34))
    p.poly([(-11, -7.2), (11, -7.2), (10, -4), (-10, -4)], c["eye"], ol=False)
    p.ell(6, -5.6, 2.5, 1.3, (255, 200, 200), ol=False)
    for y in (7, 10, 13):
        p.line([(-6, y), (6, y)], (70, 76, 96), w=1.1)
    p.line([(0, -26), (0, -12)], sh(f, 0.6), w=1.5)
    p.ell(-7, -18, 4, 2.5, sh(f, 1.3), ol=False)


def ead(p, c, opened=False):
    p.blob(0, 0, 25, 25, c["suit"])
    p.blob(0, -1, 20, 20, c["fur"])
    for s in (-1, 1):
        p.poly([(s * 3, -6), (s * 11, -9), (s * 10, -4.5), (s * 4, -3.6)], (30, 20, 20))
        p.ell(s * 7.5, -6, 1, 0.9, (255, 250, 200), ol=False)
    p.poly([(-15, 3), (-7, 6), (7, 6), (15, 3), (10, 13), (0, 16), (-10, 13)], (90, 30, 20), smooth=2)
    p.poly([(-12, 5.5), (12, 5.5), (10, 8.5), (-10, 8.5)], WHITE, ol=False)
    p.poly([(0, 1), (-7, -1), (-15, 2), (-9, 4.5), (0, 3.5), (9, 4.5), (15, 2), (7, -1)], (110, 60, 20), smooth=1)
    p.ell(-7, -14, 5, 3, c["light"], ol=False)


def octo(p, c, opened=False):
    f = c["fur"]
    for i, x in enumerate((-18, -10, 10, 18)):
        p.poly([(x - 4, 4), (x + 4, 4), (x + 3 + (i - 1.5) * 2, 26), (x - 1 + (i - 1.5) * 2, 28)], sh(f, 0.85),
               smooth=1)
    p.blob(0, -8, 22, 22, f)
    for x in (-8, 8):
        p.ell(x, -2, 5.5, 4, (250, 250, 240))
        p.ell(x, -1.4, 3.8, 1.4, c["eye"], ol=False)
        p.poly([(x - 6, -8), (x + 6, -5), (x + 6, -3.6), (x - 6, -6.2)] if x < 0 else
               [(x + 6, -8), (x - 6, -5), (x - 6, -3.6), (x + 6, -6.2)], sh(f, 0.55), ol=False)
    p.blob(0, 11, 5, 6, sh(f, 1.05))
    p.ell(0, 14, 2.6, 2.4, (60, 10, 20))
    for (x, y) in [(-10, -20), (6, -22), (14, -12)]:
        p.ell(x, y, 2, 1.6, c["light"], ol=False)


def alien(p, c, opened=False):
    f = c["fur"]
    p.poly([(-19, 14), (-22, -8), (-14, -26), (0, -32), (14, -26), (22, -8), (19, 14), (8, 22), (-8, 22)],
           sh(f, 0.75), smooth=2)
    p.poly([(-17, 12), (-19, -8), (-12, -24), (0, -29), (12, -24), (17, -8), (15, 12), (6, 19), (-8, 19)], f,
           ol=False, smooth=2)
    for x in (-8, 0, 8):
        p.line([(x * 1.1, -30), (x * 1.5, -12)], sh(f, 0.55), w=1.3, smooth=1)
    for s in (-1, 1):
        p.poly([(s * 3, -5), (s * 13, -8), (s * 13, -1), (s * 4, 1)], c["eye"], smooth=1)
        p.ell(s * 8, -3.4, 1.4, 2.2, INK[:3], ol=False)
        p.poly([(s * 2, -8), (s * 14, -11), (s * 14, -9), (s * 3, -6)], sh(f, 0.5), ol=False)
    p.ell(-1.4, 6, 0.8, 1.1, INK[:3], ol=False)
    p.ell(1.4, 6, 0.8, 1.1, INK[:3], ol=False)
    p.line([(-8, 13), (-3, 11.5), (3, 11.5), (8, 13)], w=1.3, smooth=1)
    p.ell(-9, -20, 4, 2.5, c["light"], ol=False)


def skull(p, c, opened=False):
    b = c["fur"]
    p.blob(0, -6, 19, 20, b)
    p.poly([(-13, 4), (13, 4), (11, 18), (-11, 18)], sh(b, 0.9), smooth=1)
    for x in (-7, 7):
        p.ell(x, -2, 5.2, 5, (26, 16, 24))
        p.ell(x, -1.5, 1.4, 1.4, c["eye"], ol=False)
    p.poly([(0, 3), (2.4, 8), (-2.4, 8)], (26, 16, 24))
    p.poly([(-10, 11), (10, 11), (9, 18), (-9, 18)], (30, 20, 26))
    for i in range(6):
        x = -8 + i * 3.2
        p.poly([(x - 1.3, 11.4), (x + 1.3, 11.4), (x + 1.2, 14), (x - 1.2, 14)], WHITE, ol=False)
        p.poly([(x - 1.2, 15), (x + 1.2, 15), (x + 1.3, 17.6), (x - 1.3, 17.6)], WHITE, ol=False)
    p.line([(-8, -18), (-3, -14), (-4, -10)], sh(b, 0.6), w=0.8)


def draq(p, c, opened=False):
    f = c["fur"]
    p.poly([(-28, 14), (-30, -6), (-20, -22), (0, -26), (20, -22), (30, -6), (28, 14), (14, 22), (-14, 22)],
           sh(f, 0.75), smooth=2)
    p.poly([(-25, 10), (-27, -6), (-18, -20), (0, -23), (18, -20), (25, -6), (22, 10), (10, 18), (-12, 18)], f,
           ol=False, smooth=2)
    for s in (-1, 1):
        p.poly([(s * 4, -10), (s * 15, -16), (s * 22, -9), (s * 12, -8)], sh(f, 0.55), ol=False)
        p.ell(s * 11, -5, 4, 2.6, WHITE)
        p.ell(s * 11, -4.8, 1.2, 1.6, INK[:3], ol=False)
    p.poly([(-18, 5), (18, 5), (12, 15), (-12, 15)], (80, 10, 40), smooth=1)
    for x in (-12, -5, 5, 12):
        p.poly([(x - 2, 5.5), (x + 2, 5.5), (x, 10.5)], WHITE, ol=False)
    p.ell(-12, -16, 5, 2.5, c["light"], ol=False)


def twin(p, c, opened=False):
    f = c["fur"]
    # Shioh: small head at upper right, behind
    p.at(p.ox + p.k * 16, p.oy - p.k * 6, p.k * 0.55)
    draw_face(p, dict(skin=(200, 150, 120), jaw=0.95))
    for x in (-7, 7):
        eye(p, x, -1, (40, 40, 40), narrow=True)
    brows(p, dict(hair=(60, 40, 40)), "angry")
    mouth(p, dict(skin=(200, 150, 120)), "closed")
    p.poly([(-17, -2), (-17, -18), (0, -24), (17, -18), (17, -2), (0, -12)], (150, 60, 170), smooth=1)
    p.at(p.ox - p.k / 0.55 * 16, p.oy + p.k / 0.55 * 6, p.k / 0.55)
    # Gomar: broad heavy head in front, left
    p.at(p.ox - p.k * 5, p.oy + p.k * 2, p.k)
    p.poly([(-20, 18), (-22, -4), (-16, -20), (0, -24), (14, -20), (18, -4), (16, 18), (0, 24)], sh(f, 0.72),
           smooth=2)
    p.poly([(-18, 16), (-20, -4), (-14, -18), (0, -21), (12, -18), (15, -4), (13, 15), (0, 20)], f, ol=False,
           smooth=2)
    p.poly([(-19, -9), (15, -9), (13, -4), (-17, -4)], sh(f, 0.5), ol=False)
    p.ell(-6, -1, 3.8, 2.2, (250, 230, 120))
    p.ell(-6, -1, 1.2, 1.6, INK[:3], ol=False)
    p.ell(7, -1, 3.4, 1.4, sh(f, 0.45))
    p.poly([(0, 1), (3, 8), (-3, 8)], sh(f, 0.6))
    p.poly([(-9, 12), (9, 12), (6, 16), (-6, 16)], (240, 230, 220))
    p.line([(-8, 14), (8, 14)], w=0.6)
    p.ell(-8, -14, 5, 2.5, c["light"], ol=False)


def dog(p, c, opened=False):
    f, l = c["fur"], c["light"]
    for s in (-1, 1):
        p.poly([(s * 8, -14), (s * 20, -34), (s * 21, -8)], f)
        p.poly([(s * 11, -15), (s * 19, -28), (s * 19, -11)], (230, 170, 170), ol=False)
    p.blob(0, -4, 20, 20, f)
    p.poly([(-4, -24), (4, -24), (6, -6), (14, 4), (12, 18), (0, 22), (-12, 18), (-14, 4), (-6, -6)], l, smooth=1)
    for x in (-8, 8):
        eye(p, x, -6, (40, 90, 160), mirror=x > 0)
        s = 1 if x > 0 else -1
        p.poly([(x - s * 4, -10), (x + s * 4, -12), (x + s * 4, -10.5), (x - s * 4, -9)], sh(f, 0.5), ol=False)
    p.ell(0, 5, 3.6, 2.6, (20, 20, 24))
    p.ell(-1, 4.2, 1, 0.7, (150, 150, 160), ol=False)
    p.line([(0, 7.5), (0, 11)], w=0.9)
    p.line([(-6, 11), (-3, 13), (0, 11), (3, 13), (6, 11)], w=1.0, smooth=1)


def zero(p, c, opened=False):
    s, hc = c["skin"], c["hair"]
    draw_ears(p, dict(skin=s))
    draw_face(p, dict(skin=s, jaw=1.05))
    p.poly([(-18, 0), (-19, -14), (-10, -24), (4, -27), (16, -20), (19, -8), (17, 0), (14, -9), (4, -12), (-8, -11),
            (-15, -7)], hc, smooth=2)
    p.poly([(-17, -5), (17, -5), (16, 2), (-16, 2)], (40, 40, 60))
    for x in (-7, 7):
        p.ell(x, -1.5, 4.6, 3.6, (30, 20, 40))
        p.ell(x, -1.5, 3, 2.3, c["eye"], ol=False)
        p.ell(x - 0.8, -2.3, 1, 0.8, (255, 230, 240), ol=False)
    nose(p, dict(skin=s))
    p.poly([(-8, 9), (-2, 7.5), (0, 8.5), (2, 7.5), (8, 9), (5, 10.5), (-5, 10.5)], (30, 30, 70), smooth=1)
    if opened:
        p.poly([(-6, 11), (6, 11), (3.5, 18), (-3.5, 18)], (70, 10, 30), smooth=1)
        p.poly([(-5, 11.4), (5, 11.4), (4.5, 13), (-4.5, 13)], WHITE, ol=False)
    else:
        p.line([(-6, 13), (0, 14), (6, 13)], w=1.2, smooth=1)


TEMPLATES = {"human": human, "monkey": monkey, "lizard": lizard, "rex": rex, "robot": robot, "ead": ead,
             "octo": octo, "alien": alien, "skull": skull, "draq": draq, "twin": twin, "dog": dog, "zero": zero}


def shoulders(p, c):
    suit, col = c.get("suit", (80, 80, 100)), c.get("collar", (230, 230, 230))
    if c["t"] == "rex":
        p.blob(12, 44, 30, 20, suit)
        return
    p.poly([(-40, 60), (-36, 30), (-22, 22), (22, 22), (36, 30), (40, 60)], sh(suit, 0.72), smooth=1)
    p.poly([(-37, 60), (-33, 31), (-21, 24.5), (18, 24.5), (31, 31), (34, 60)], suit, ol=False, smooth=1)
    for s in (-1, 1):
        p.ell(s * 30, 32, 9, 7, col)
    if c["t"] in ("human", "zero", "monkey", "dog"):
        p.poly([(-7, 16), (7, 16), (8, 26), (-8, 26)], sh(c.get("skin", c.get("face", c.get("fur"))), 0.7))
    p.poly([(-13, 21), (-6, 27), (0, 24), (6, 27), (13, 21), (11, 32), (0, 36), (-11, 32)], col, smooth=1)


def head(p, c, opened=False):
    sc = c.get("scale", 1.0)
    if sc != 1.0:
        p.at(p.ox, p.oy, p.k * sc, p.fx < 0)
    TEMPLATES[c["t"]](p, c, opened)


def brief(name):
    return CHARACTERS[ALIASES.get(name, name)]


# ------------------------------------------------------------------ pictures
def bust(w, h, c, cx, cy, k, bg=None, opened=False, ol=0.85):
    p = Pen(w, h, ol=ol, bg=bg)
    p.at(cx, cy, k)
    shoulders(p, c)
    p.at(cx, cy, k)
    head(p, c, opened)
    return p.result()


def frame(img, col=(245, 245, 245), inner=(16, 14, 20)):
    img = img.copy()
    img[0, :, :3] = img[-1, :, :3] = col
    img[:, 0, :3] = img[:, -1, :3] = col
    img[1, 1:-1, :3] = img[-2, 1:-1, :3] = inner
    img[1:-1, 1, :3] = img[1:-1, -2, :3] = inner
    img[..., 3] = 255
    return img


def backdrop(w, h, c):
    bg = np.array(c.get("bg", (30, 30, 40)), np.float32)
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.empty((h, w, 4), np.float32)
    img[..., :3] = bg * (1.2 - 0.7 * y)
    img[..., 3] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


def portrait(name, w):
    c = brief(name)
    bg = backdrop(w, w, c)
    k = w / 50.0
    fg = bust(w, w, c, w * 0.5, w * 0.47, k, ol=0.75 if w <= 32 else 0.85)
    return frame(over(bg, fg))


def over(bg, fg):
    a = fg[..., 3:4].astype(np.float32) / 255.0
    out = bg.astype(np.float32).copy()
    out[..., :3] = fg[..., :3] * a + out[..., :3] * (1 - a)
    out[..., 3] = np.maximum(out[..., 3], fg[..., 3])
    return np.clip(out, 0, 255).astype(np.uint8)


def grid_rgb(e):
    g = np.asarray(e["grid"], np.float32)
    n = int(round(len(g) ** 0.5))
    g = g.reshape(n, n, 4)
    a = g[..., 3:4] / 255.0
    rgb = np.where(a > 0.02, np.clip(g[..., :3] / np.maximum(a, 1e-3), 0, 255), 0)
    return rgb, g[..., 3]


def ending(name, e):
    c = brief(name)
    w, h = e["w"], e["h"]
    rgb, _ = grid_rgb(e)
    im = Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6))
    bg = np.asarray(im, np.float32) * 0.55 + np.array(c.get("bg", (30, 30, 40)), np.float32) * 0.45
    # speed lines behind the pilot
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(yy - h * 0.45, xx - w * 0.5)
    rays = (np.sin(ang * 18) > 0.6) * 0.18
    bg = bg * (1 + rays[..., None])
    bg = np.dstack([np.clip(bg, 0, 255), np.full((h, w), 255.0)]).astype(np.uint8)
    k = h / 58.0
    say = BUBBLES.get(name)
    fg = bust(w, h, c, w * (0.64 if say else 0.5), h * 0.42, k, ol=1.0)
    out = over(bg, fg)
    if say:
        out = over(out, bubble(w, h, say))
    return frame(out, inner=(30, 26, 34))


# ending panels where the pilot speaks: the words are text facts, re-typeset in our font
BUBBLES = {"CaptainFalconAlt": "TOP SPOT!", "JodySummerAlt": "CHEERS!", "SamuraiGorohAlt": "NOT BAD!"}


def bubble(w, h, text):
    from . import labels
    im = Image.new("RGBA", (w * 4, h * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy, rx, ry = w * 0.25 * 4, h * 0.4 * 4, w * 0.2 * 4, h * 0.2 * 4
    d.polygon([(cx + rx * 0.5, cy + ry * 0.6), (cx + rx * 1.35, cy + ry * 0.9), (cx + rx * 0.8, cy + ry * 0.2)],
              fill=(250, 250, 250, 255), outline=INK, width=4)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(250, 250, 250, 255), outline=INK, width=5)
    d.polygon([(cx + rx * 0.55, cy + ry * 0.5), (cx + rx * 1.3, cy + ry * 0.86), (cx + rx * 0.75, cy + ry * 0.25)],
              fill=(250, 250, 250, 255))
    px = 40
    while px > 8:
        f = labels.font("menu", px)
        b = d.textbbox((0, 0), text, font=f)
        if b[2] - b[0] <= rx * 1.6:
            break
        px -= 2
    d.text((cx - (b[2] + b[0]) / 2, cy - (b[3] + b[1]) / 2), text, font=f, fill=INK)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.uint8)


def full(name, e):
    from scipy import ndimage
    from cleanroom.decomp.gen import unpack_alpha2
    c = brief(name)
    w, h = e["w"], e["h"]
    # our own standing pilot (2026-10 rebuild: no retail silhouette or colours): suit body, trim bands,
    # darker gloves and boots, drawn from the brief's suit / collar colours; pose varies per pilot
    rng = np.random.default_rng(len(name) * 7 + sum(map(ord, name)))
    S = 4
    lab_im = Image.new("L", (w * S, h * S), 0)
    dr = ImageDraw.Draw(lab_im)
    cx, top = w * S / 2, h * S * 0.13
    sw = w * S * rng.uniform(0.30, 0.38)           # shoulder half-width
    hip = sw * rng.uniform(0.55, 0.7)
    yb, yw, yk, yf = h * S * 0.24, h * S * 0.55, h * S * 0.76, h * S * 0.97
    stance = rng.uniform(0.25, 0.55)
    torso = [(cx - sw, yb), (cx + sw, yb), (cx + hip, yw), (cx - hip, yw)]
    dr.polygon(torso, fill=1)
    for sgn in (-1, 1):
        leg = [(cx + sgn * hip * 0.15, yw), (cx + sgn * hip, yw), (cx + sgn * hip * (1 + stance), yk),
               (cx + sgn * hip * (1.1 + stance), yf), (cx + sgn * hip * (0.35 + stance), yf),
               (cx + sgn * hip * 0.25, yk)]
        dr.polygon(leg, fill=1)
        dr.polygon([(cx + sgn * hip * (1.1 + stance), yf), (cx + sgn * hip * (0.25 + stance), yf),
                    (cx + sgn * hip * (0.3 + stance), yf - h * S * 0.06), (cx + sgn * hip * (1.05 + stance), yf - h * S * 0.06)],
                   fill=3)
        arm = [(cx + sgn * sw, yb), (cx + sgn * sw * 1.25, yb + h * S * 0.05),
               (cx + sgn * sw * 1.15, yw * 0.98), (cx + sgn * sw * 0.9, yw * 0.98), (cx + sgn * sw * 0.85, yb + h * S * 0.08)]
        dr.polygon(arm, fill=1)
        dr.ellipse([cx + sgn * sw * 1.03 - sw * 0.16, yw * 0.95, cx + sgn * sw * 1.03 + sw * 0.16, yw * 1.08], fill=3)
    dr.rectangle([cx - hip, yw - h * S * 0.035, cx + hip, yw], fill=2)
    dr.polygon([(cx - sw * 0.35, yb), (cx + sw * 0.35, yb), (cx + sw * 0.12, yb + h * S * 0.12),
                (cx - sw * 0.12, yb + h * S * 0.12)], fill=2)
    dr.rectangle([cx - sw * 0.17, top + h * S * 0.07, cx + sw * 0.17, yb + 2], fill=1)
    lab = np.asarray(lab_im.resize((w, h), Image.NEAREST), np.int32)
    m = lab > 0
    suit = np.array(c.get("suit", (80, 80, 120)), np.float32)
    trim = np.array(c.get("collar", (230, 230, 230)), np.float32)
    cen = np.stack([suit, suit, trim, suit * 0.45 + 10])
    base = cen[lab]
    # body shading from the silhouette: pseudo-normals from a smoothed distance field, light from upper left
    d = ndimage.distance_transform_edt(m)
    hgt = ndimage.gaussian_filter(np.sqrt(d), 2.0)
    gy, gx = np.gradient(hgt)
    nz = np.ones_like(hgt) * 0.6
    L = np.array([-0.6, -0.5, 0.62])
    nn = np.sqrt(gx ** 2 + gy ** 2 + nz ** 2)
    lit = (-gx * L[0] - gy * L[1] + nz * L[2]) / nn
    tone = np.where(lit > 0.85, 1.22, np.where(lit > 0.62, 1.05, np.where(lit > 0.3, 0.88, 0.62)))
    col = np.clip(base * tone[..., None], 0, 255)
    # outlines: silhouette edge + boundaries between colour regions
    edge = m & ~ndimage.binary_erosion(m, iterations=1)
    seam = np.zeros_like(m)
    seam[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    seam[1:, :] |= lab[1:, :] != lab[:-1, :]
    col[edge | (seam & m)] = INK[:3]
    img = np.dstack([col, np.where(m, 255.0, 0.0)])
    # head: on our figure's neck, sized from the shoulders
    hx = w / 2
    kk = (sw / S) * 0.62 / 16.0
    hy = (top + h * S * 0.09) / S - 21 * kk
    p = Pen(w, h, ol=0.9)
    p.at(hx, hy, kk)
    head(p, c)
    hd = p.result()
    out = over(img.astype(np.uint8), hd)
    # keep the silhouette alpha, but let the head poke out where our drawing is fuller
    out[..., 3] = np.where(m | (hd[..., 3] > 128), 255, 0)
    return out


def mrzero(e, opened):
    from cleanroom.decomp.gen import unpack_alpha2
    c = CHARACTERS["MrZero"]
    w, h = e["w"], e["h"]
    bg = np.zeros((h, w, 4), np.uint8)
    yy = np.arange(h)[:, None]
    scan = ((yy % 3) == 0) * 18.0
    b = np.array(c["bg"], np.float32)
    bg[..., :3] = np.clip(b * (1.25 - 0.6 * yy[..., None] / h) - scan[..., None], 0, 255)
    bg[..., 3] = 255
    fg = bust(w, h, c, w * 0.5, h * 0.44, w / 52.0, opened=opened, ol=0.75)
    return over(bg, fg)


RX = re.compile(r"^a(Portrait|SmallPortrait|FullPortrait|Ending)(\w+?)Tex$")


def hook(sym, e):
    if sym == "aCountdownMrZeroMouthClosedTex":
        return mrzero(e, False)
    if sym == "aCountdownMrZeroMouthOpenTex":
        return mrzero(e, True)
    m = RX.match(sym)
    if not m:
        return None
    kind, name = m.groups()
    if ALIASES.get(name, name) not in CHARACTERS or name == "MrZero":
        return None
    if kind in ("Portrait", "SmallPortrait"):
        return portrait(name, e["w"])
    if kind == "FullPortrait":
        return full(name, e)
    if kind == "Ending":
        return ending(name, e)
    return None


if __name__ == "__main__":
    import sys
    tiles = [portrait(n, 40) for n in CHARACTERS]
    sheet = np.zeros((48 * 4, 48 * 9, 4), np.uint8)
    for i, t in enumerate(tiles):
        y, x = divmod(i, 9)
        sheet[y * 48 + 4:y * 48 + 44, x * 48 + 4:x * 48 + 44] = t
    Image.fromarray(sheet).resize((sheet.shape[1] * 3, sheet.shape[0] * 3), Image.NEAREST).save(sys.argv[1])
