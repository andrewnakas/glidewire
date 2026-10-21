"""The six glyph fonts (one texture per glyph: aFont<N><Name>Tex): our own glyphs from OFL fonts.

The character comes from the symbol name; the glyph is fitted into the box of the kept alpha outline (so the
game's per-glyph spacing still works) and coloured from the kept grid (Font1: vertical gradient; I4 fonts are
white, the game tints them). Styles: 1 heavy italic, 2 tiny pixel, 3 bold italic, 4 small pixel, 5 italic words,
6 bold.
"""
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cleanroom.decomp.gen import unpack_alpha2

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
SS = 8

NAMES = {
    "Ampersand": "&", "Comma": ",", "Dash": "-", "Period": ".", "Prime": "'", "DoublePrime": '"',
    "ExclamationMark": "!", "Underscore": "_", "Colon": ":", "Ellipsis": "...", "Alpha": "α",
    "Beta": "β", "Gamma": "γ", "Box": "□", "End": "END", "Kg": "Kg", "Minus": "-", "Plus": "+",
    "Points": "PTS.", "ND": "nd", "RD": "rd", "ST": "st", "TH": "th", "LowerC": "c", "LowerR": "r",
    "LowerS": "s", "CloseParenthesis": ")", "OpenParenthesis": "(", "Computer": "Computer", "Player": "Player",
    "Quit": "Quit", "SelectCourse": "Select Course", "DownArrow": "▼", "UpArrow": "▲",
    "SideArrow": "▶", "HiraganaNi": "に",
}
KATAKANA = {
    "A": "ア", "I": "イ", "U": "ウ", "E": "エ", "O": "オ", "Ka": "カ", "Ki": "キ", "Ku": "ク", "Ke": "ケ", "Ko": "コ",
    "Ga": "ガ", "Gi": "ギ", "Gu": "グ", "Ge": "ゲ", "Go": "ゴ", "Sa": "サ", "Shi": "シ", "Su": "ス", "Se": "セ",
    "So": "ソ", "Za": "ザ", "Ji": "ジ", "Zu": "ズ", "Ze": "ゼ", "Zo": "ゾ", "Ta": "タ", "Chi": "チ", "Tsu": "ツ",
    "Te": "テ", "To": "ト", "Da": "ダ", "Dji": "ヂ", "Dzu": "ヅ", "De": "デ", "Do": "ド", "Na": "ナ", "Ni": "ニ",
    "Nu": "ヌ", "Ne": "ネ", "No": "ノ", "Ha": "ハ", "Hi": "ヒ", "Fu": "フ", "He": "ヘ", "Ho": "ホ", "Ba": "バ",
    "Bi": "ビ", "Bu": "ブ", "Be": "ベ", "Bo": "ボ", "Pa": "パ", "Pi": "ピ", "Pu": "プ", "Pe": "ペ", "Po": "ポ",
    "Ma": "マ", "Mi": "ミ", "Mu": "ム", "Me": "メ", "Mo": "モ", "Ya": "ヤ", "Yu": "ユ", "Yo": "ヨ", "Ra": "ラ",
    "Ri": "リ", "Ru": "ル", "Re": "レ", "Ro": "ロ", "Wa": "ワ", "Wo": "ヲ", "N": "ン", "YoonYa": "ャ",
    "YoonYu": "ュ", "YoonYo": "ョ", "DigraphA": "ァ", "DigraphI": "ィ", "DigraphE": "ェ", "Soukon": "ッ",
}
STYLE = {  # font file, variation, italic shear
    "1": ("Exo2-Italic[wght].ttf", 900, 0.0),
    "2": ("PressStart2P-Regular.ttf", None, 0.0),
    "3": ("Exo2-Italic[wght].ttf", 800, 0.0),
    "4": ("Rubik.ttf", 700, 0.0),
    "5": ("Exo2-Italic[wght].ttf", 700, 0.0),
    "6": ("RussoOne-Regular.ttf", None, 0.18),
}
RX = re.compile(r"^aFont([1-6])(Katakana)?(\w+?)Tex$")


def char_of(sym):
    m = RX.match(sym)
    if not m:
        return None, None
    n, kata, name = m.groups()
    if kata:
        return n, KATAKANA.get(name)
    if name.startswith("Num") and name[3:].isdigit():
        return n, name[3:]
    if name in NAMES:
        return n, NAMES[name]
    if len(name) == 1:
        return n, name
    return n, None


def _font(n, text, px):
    fn, var, _ = STYLE[n]
    if str(n) == "4":
        # the 8 px menu font: square pixel letters, thin kana (our style; the gate found rounded bold too close)
        fn, var = ("NotoSansJP-Regular.ttf", None) if any(ord(c) > 0x2E80 for c in text) else \
            ("PressStart2P-Regular.ttf", None)
    elif any(ord(c) > 0x2E80 for c in text):
        fn, var = "MPLUSRounded1c-ExtraBold.ttf", None
    elif any(c in text for c in "αβγ□▼▲▶"):
        fn, var = "NotoSansJP-Regular.ttf", None
    f = ImageFont.truetype(os.path.join(FONTS, fn), max(4, int(px)))
    if var:
        try:
            f.set_variation_by_axes([var])
        except Exception:
            pass
    return f


# advance widths of font set 4's capitals (code table kFont4CharWidths in ovl_i2/font.c: structure)
FONT4_WIDTHS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        [7, 7, 6, 7, 6, 5, 6, 7, 4, 4, 6, 5, 7, 6, 7, 6, 7, 7, 6, 6, 7, 6, 7, 7, 7, 5]))


def glyph(n, text, e):
    w, h = e["w"], e["h"]
    if "alpha2" in e:
        a = unpack_alpha2(e["alpha2"], w, h)
    else:
        a = np.full((h, w), 255.0)
    ys, xs = np.nonzero(a > 40)
    if len(xs) == 0:
        return np.zeros((h, w, 4), np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if str(n) == "4" and w == 16:
        # font set 4 cells are 16 wide but drawn 8 wide (font.c CACHE_TEX_INFO_WH(..., 8, ...)) and advanced by
        # kFont4CharWidths: keep the glyph inside the left part of the cell
        adv = FONT4_WIDTHS.get(text, 8)
        x0, x1 = 0, max(3, adv - 1)
    bw, bh = (x1 - x0) * SS, (y1 - y0) * SS
    W, H = w * SS, h * SS
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    lo, hi, best = 4, H * 2, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = _font(n, text, mid)
        b = d.textbbox((0, 0), text, font=f)
        if b[2] - b[0] <= bw * 1.02 and b[3] - b[1] <= bh * 1.02:
            best, lo = (f, b), mid + 1
        else:
            hi = mid - 1
    if best is None:
        f = _font(n, text, 4)
        best = (f, d.textbbox((0, 0), text, font=f))
    f, b = best
    cx, cy = (x0 + x1) / 2 * SS, (y0 + y1) / 2 * SS
    d.text((cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2), text, font=f, fill=255)
    shear = STYLE[n][2]
    if shear:
        img = img.transform(img.size, Image.AFFINE, (1, shear, -shear * H / 2, 0, 1, 0), Image.BILINEAR)
    m = np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0
    if n in ("2", "4"):
        m = np.clip((m - 0.25) * 2.0, 0, 1)        # crisp small glyphs
    rgb = np.full((h, w, 3), 255.0, np.float32)
    if e["fmt"] == "RGBA16":
        g = np.asarray(e["grid"], np.float32)
        k = int(round(len(g) ** 0.5))
        g = g.reshape(k, k, 4)
        al = g[..., 3:4] / 255.0
        ok = al[..., 0] > 0.15
        if ok.any():
            c = np.clip(g[..., :3] / np.maximum(al, 1e-3), 0, 255)
            rows = [r for r in range(k) if ok[r].any()]
            top = c[rows[0]][ok[rows[0]]].max(0)
            bot = c[rows[-1]][ok[rows[-1]]].max(0)
            t = np.linspace(0, 1, h)[:, None, None]
            rgb = np.broadcast_to(top[None, None] * (1 - t) + bot[None, None] * t, (h, w, 3)).copy()
            rgb = rgb * (235.0 / max(rgb.max(), 1.0))
    out = np.dstack([rgb, m * 255.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def hook(sym, e):
    n, text = char_of(sym)
    if not text:
        return None
    return glyph(n, text, e)
