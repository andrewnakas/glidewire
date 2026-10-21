"""2D tile-level originality (NES/SNES/GB platformers): a generated level must not
reproduce a retail level's layout.

Input: JSON {"grid": [[class, ...], ...]} rows x columns, 0 = empty, other ints =
classes (solid, brick, block, coin, pipe, platform, enemy, ...).

    compare(retail_grid, clean_grid, cfg) -> dict(score, feature_iou, hole_jaccard, fail)

- Base bands (rows filled across >= band_fill of the width in a grid: the default
  floor / ceiling) are structure, not layout, and are left out of the features.
- feature_iou: same-class cell IoU of the remaining cells, best over horizontal
  shifts of +-max_shift columns, on the common width.
- hole_jaccard: IoU of the columns where the base floor is missing (pits): clean pit
  columns within 1 column of a retail pit, over the union of both pit sets; counted
  only when both levels have >= min_holes.
- structure_classes (cfg): cell classes that are structure (e.g. terrain fill): never
  features; pits are then columns with no structure cell in the lower half.
- anchors: clean columns hard-wired by the code, blanked on both sides.
- score = max of the two; fail if score > iou_max.
"""
import numpy as np

from cleanroom.originality.config import CONFIG


def _bands(g, fill):
    return (g != 0).mean(axis=1) >= fill


def _features(g, fill, structure=()):
    f = g.copy()
    f[_bands(g, fill)] = 0
    for k in structure:
        f[f == k] = 0
    return f


def _holes(g, fill, structure=()):
    if structure:                                  # pits = columns without floor-structure cells
        m = np.isin(g, structure)
        floor = m[g.shape[0] // 2:].any(axis=0)
        return ~floor & (m.any(axis=1).sum() > 0)
    rows = np.nonzero(_bands(g, fill))[0]
    rows = rows[rows >= g.shape[0] // 2]           # floor bands only
    if not len(rows):
        return np.zeros(g.shape[1], bool)
    return (g[rows] == 0).all(axis=0)


def compare(retail, clean, cfg=None, anchors=()):
    """anchors: clean columns the code hard-wires (excluded on both sides)"""
    cfg = cfg or CONFIG["tilemap"]
    r, c = np.array(retail), np.array(clean)
    for x in anchors:
        if x < r.shape[1]:
            r[:, x] = 0
        if x < c.shape[1]:
            c[:, x] = 0
    h = min(r.shape[0], c.shape[0])
    r, c = r[:h], c[:h]
    st = cfg.get("structure_classes", [])
    fr, fc = _features(r, cfg["band_fill"], st), _features(c, cfg["band_fill"], st)
    best = 0.0
    for s in range(-cfg["max_shift"], cfg["max_shift"] + 1):
        a = fr[:, max(0, s):]
        b = fc[:, max(0, -s):]
        w = min(a.shape[1], b.shape[1])
        a, b = a[:, :w], b[:, :w]
        union = ((a != 0) | (b != 0)).sum()
        if union:
            best = max(best, float(((a == b) & (a != 0)).sum()) / union)
    hr, hc = _holes(r, cfg["band_fill"], st), _holes(c, cfg["band_fill"], st)
    w = min(len(hr), len(hc))
    hr, hc = hr[:w].copy(), hc[:w].copy()
    for x in anchors:
        if x < w:
            hr[x] = hc[x] = False
    hj = 0.0
    if hr.sum() >= cfg["min_holes"] and hc.sum() >= cfg["min_holes"]:
        dil = lambda v: v | np.roll(v, 1) | np.roll(v, -1)
        hj = float((hc & dil(hr)).sum()) / float((hr | hc).sum())
    score = max(best, hj)
    return dict(score=round(score, 3), feature_iou=round(best, 3), hole_jaccard=round(hj, 3),
                fail=score > cfg["iou_max"])
