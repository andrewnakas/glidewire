"""Image originality: perceptual hashes + SSIM of retail vs clean textures.

Both images are composited on mid grey by alpha, converted to luma and
resized to `size` (bilinear). Flat retail images (luma std < flat_std) are
structural (a fill colour) and skipped.

  dhash   64-bit gradient hash (9x8), Hamming distance
  phash   64-bit DCT hash (8x8 low frequencies of the working image)
  ssim    mean SSIM over 7x7 windows of the working image

Fail: ssim > ssim_max, or (phash distance <= phash_min_dist and dhash
distance <= dhash_min_dist).

    compare(retail_rgba, clean_rgba, cfg=None) -> dict
"""
import numpy as np
from scipy.fft import dctn
from scipy.ndimage import uniform_filter

from cleanroom.originality.config import CONFIG


def luma(img):
    a = np.asarray(img, np.float64)
    if a.ndim == 2:
        return a
    rgb = a[..., :3]
    if a.shape[-1] == 4:
        al = a[..., 3:4] / 255.0
        rgb = rgb * al + 128.0 * (1 - al)
    return rgb @ np.array([0.299, 0.587, 0.114])


def resize(g, w, h):
    H, W = g.shape
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
    bot = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def dhash(g):
    s = resize(g, 9, 8)
    bits = (s[:, 1:] > s[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash(g):
    d = dctn(resize(g, 32, 32), norm="ortho")[:8, :8].ravel()
    med = np.median(d[1:])
    return int("".join("1" if v > med else "0" for v in d), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def ssim(a, b, win=7):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a ** 2
    vb = uniform_filter(b * b, win) - mu_b ** 2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def compare(retail, clean, cfg=None):
    cfg = cfg or CONFIG["images"]
    r, c = luma(retail), luma(clean)
    if r.std() < cfg["flat_std"]:
        return {"flat": True, "fail": False, "why": [], "score": 0.0}
    n = cfg["size"]
    rs, cs = resize(r, n, n), resize(c, n, n)
    s = ssim(rs, cs)
    pd, dd = hamming(phash(r), phash(c)), hamming(dhash(r), dhash(c))
    why = []
    if s > cfg["ssim_max"]:
        why.append(f"SSIM {s:.2f} > {cfg['ssim_max']}")
    if pd <= cfg["phash_min_dist"] and dd <= cfg["dhash_min_dist"]:
        why.append(f"pHash {pd} / dHash {dd} bits apart")
    return {"ssim": round(s, 4), "phash": pd, "dhash": dd, "flat": False,
            "score": round(max(s / cfg["ssim_max"], (64 - pd) / (64 - cfg["phash_min_dist"]) if dd <= cfg["dhash_min_dist"] else 0), 4),
            "fail": bool(why), "why": why}
