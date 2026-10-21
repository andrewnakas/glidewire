"""Pictures of machines and people in the staff credits, drawn as clean cel-shaded shapes.

aCredits<Machine>Tex (30 machines, ~128x90 on black): the kept alpha silhouette filled with a few flat colours
(k-means of the kept 16x16 grid, lifted and saturated), shaded from the silhouette's own distance field (light
from the upper left), with dark outlines around the shape and between colour regions and a highlight streak.
aCreditsMrZeroTex / aCreditsMenuLadyTex: our pilot painter (pilots.py) on a dark card.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from cleanroom.decomp.gen import unpack_alpha2

from . import pilots


def cel(e, k=4, seed=0, bg=(10, 10, 14)):
    w, h = e["w"], e["h"]
    a = unpack_alpha2(e["alpha2"], w, h) if "alpha2" in e else None
    rgb, ga = pilots.grid_rgb(e)
    lum = rgb.sum(-1)
    if a is None:
        # opaque picture on black: the silhouette is where the grid is not near-black
        sm = np.asarray(Image.fromarray(lum.clip(0, 255 * 3).astype(np.float32) / 3).resize((w, h), Image.BILINEAR))
        m = sm > 28
    else:
        m = a > 100
    if m.sum() < 30:
        return None
    cells = rgb[(lum > 60)].reshape(-1, 3)
    if len(cells) < k:
        return None
    rng = np.random.default_rng(seed)
    cen = cells[rng.choice(len(cells), k, replace=False)].copy()
    for _ in range(12):
        lab = ((cells[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        for i in range(k):
            if (lab == i).any():
                cen[i] = cells[lab == i].mean(0)
    for i in range(k):
        v = cen[i] * min(1.8, 225.0 / max(float(cen[i].max()), 1.0))
        cen[i] = np.clip(v.mean() + (v - v.mean()) * 1.35, 0, 255)
    smooth = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32)
    # push colour outward from the shape so the rim does not pick up the black background
    idx = ndimage.distance_transform_edt(~(smooth.sum(-1) > 90), return_distances=False, return_indices=True)
    smooth = smooth[idx[0], idx[1]]
    lab = ((smooth[..., None, :] - cen[None, None]) ** 2).sum(-1).argmin(-1)
    lab = ndimage.median_filter(lab, size=5)
    base = cen[lab]
    d = ndimage.distance_transform_edt(m)
    hgt = ndimage.gaussian_filter(np.sqrt(d), 1.5)
    gy, gx = np.gradient(hgt)
    L = np.array([-0.55, -0.6, 0.58])
    nz = np.full_like(hgt, 0.5)
    nn = np.sqrt(gx ** 2 + gy ** 2 + nz ** 2)
    shade = (gx * L[0] + gy * L[1] + nz * L[2]) / nn
    tone = np.where(shade > 0.62, 1.12, np.where(shade > 0.35, 1.0, 0.72))
    col = np.clip(base * tone[..., None], 0, 255)
    # outlines: silhouette edge and colour-region borders
    edge = m & ~ndimage.binary_erosion(m, iterations=1)
    border = (np.abs(np.diff(lab, axis=0, prepend=lab[:1])) + np.abs(np.diff(lab, axis=1, prepend=lab[:, :1]))) > 0
    col[edge | (border & m)] = (16, 14, 26)
    # highlight streak along the upper-left rim
    hl = m & (shade > 0.9) & (d > 1.5) & (d < 3.5)
    col[hl] = np.clip(col[hl] * 0.4 + 255 * 0.6, 0, 255)
    col = livery(col, m, d, cen, seed)
    out = np.zeros((h, w, 4), np.float32)
    if a is None:
        out[..., :3] = bg
        out[..., 3] = 255
        out[m, :3] = col[m]
    else:
        out[..., :3] = col
        out[..., 3] = np.where(m, 255, 0)
    return out.astype(np.uint8)


def livery(col, m, d, cen, seed):
    """Racing-machine details along the silhouette's long axis: two stripes, a canopy, glowing boosters."""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    ev, evec = np.linalg.eigh(cov)
    u = evec[:, 1]
    if u[0] < 0:
        u = -u                                   # u points to the rear (right)
    v = np.array([-u[1], u[0]])
    if v[1] > 0:
        v = -v                                   # v points up
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    t = (xx - mu[0]) * u[0] + (yy - mu[1]) * u[1]
    s_ = (xx - mu[0]) * v[0] + (yy - mu[1]) * v[1]
    tl, th = np.percentile(t[m], [2, 98])
    half = np.sqrt(ev[0]) * 1.6
    rng = np.random.default_rng(seed)
    light = cen[np.argmax(cen.sum(1))]
    accent = np.array([rng.choice([235, 40]), rng.choice([40, 210]), rng.choice([40, 230])], np.float32)
    inner = m & (d > 1.5)
    for off, c, wd in ((-0.05, np.array([245, 245, 250.0]), 1.4), (-0.3, accent, 1.1)):
        band = inner & (np.abs(s_ - off * half) < wd) & (t > tl + 0.12 * (th - tl)) & (t < th - 0.05 * (th - tl))
        col[band] = c
    # canopy: a glass ellipse on the upper side, a little ahead of the middle
    ct, cs = tl + 0.42 * (th - tl), 0.45 * half
    rt, rs = 0.13 * (th - tl), 0.32 * half
    q = ((t - ct) / max(rt, 1)) ** 2 + ((s_ - cs) / max(rs, 1)) ** 2
    glass = inner & (q < 1)
    col[glass] = np.array([90, 200, 250.0]) * 0.75 + light * 0.25
    col[inner & (q < 0.3) & (s_ > cs)] = (235, 250, 255)
    col[m & (np.abs(q - 1) < 0.25)] = (16, 14, 26)
    # boosters at the rear end
    for so in (-0.35, 0.25):
        bt, bs = th - 0.06 * (th - tl), so * half
        r = max(1.5, 0.22 * half)
        qq = (t - bt) ** 2 + (s_ - bs) ** 2
        col[m & (qq < r * r)] = (120, 240, 255)
        col[m & (qq < (r * 0.45) ** 2)] = (250, 255, 255)
    return col


def lady(e):
    """Full figure on black: the silhouette is where the kept grid is not black (no alpha outline here)."""
    w, h = e["w"], e["h"]
    rgb, _ = pilots.grid_rgb(e)
    lum = np.asarray(Image.fromarray((rgb.sum(-1) / 3).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32)
    m = ndimage.binary_opening(lum > 26, iterations=1)
    v = (m * 3).astype(np.uint8).reshape(-1)
    v = np.concatenate([v, np.zeros((-len(v)) % 4, np.uint8)]).reshape(-1, 4)
    packed = (v[:, 0] << 6) | (v[:, 1] << 4) | (v[:, 2] << 2) | v[:, 3]
    fig = pilots.full("MenuLady", dict(e, alpha2=packed.astype(np.uint8).tobytes().hex()))
    if fig is None:
        return None
    bg = np.zeros_like(fig)
    bg[..., :3], bg[..., 3] = 10, 255
    return pilots.over(bg, fig)


def card(e, name):
    w, h = e["w"], e["h"]
    c = pilots.brief(name)
    bg = pilots.backdrop(w, h, c)
    s = min(w, h)
    fg = pilots.bust(w, h, c, w / 2, h * 0.5, s / 52)
    return pilots.over(bg, fg)


def hook(sym, e):
    if not sym.startswith("aCredits") or "w" not in e:
        return None
    if sym == "aCreditsSeeYouAgainTex":
        return None
    if sym == "aCreditsMrZeroTex":
        return card(e, "MrZero")
    if sym == "aCreditsMenuLadyTex":
        return lady(e)
    return cel(e, seed=len(sym))
