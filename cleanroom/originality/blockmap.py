"""Block map originality (top-down 2D maps of blocks: Game Boy RPG towns, routes, rooms).

A clean grid must not repeat the retail grid's layout. Cells are compared at the
same position, leaving out anchor cells (warps, doors, signs the code hard-wires)
and cells where both grids show the retail background (the most common cell,
e.g. plain floor): the score is the share of the remaining cells that are equal.

    compare(retail, clean, anchors=None, cfg) -> dict(score, compared, fail)
"""
import numpy as np


def compare(retail, clean, anchors=None, cfg=None):
    cfg = cfg or {}
    r = np.asarray(retail)
    c = np.asarray(clean)
    mask = np.ones(r.shape, bool)
    for y, x in anchors or []:
        if 0 <= y < r.shape[0] and 0 <= x < r.shape[1]:
            mask[y, x] = False
    vals, counts = np.unique(r, return_counts=True)
    bg = vals[np.argmax(counts)]
    mask &= ~((r == bg) & (c == bg))
    n = int(mask.sum())
    if n < cfg.get("min_cells", 6):
        return {"score": 0.0, "compared": n, "fail": False}
    score = float((r[mask] == c[mask]).mean())
    return {"score": round(score, 3), "compared": n, "fail": score > cfg.get("equal_max", 0.35)}
