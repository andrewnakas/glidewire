"""Retail-code check: 256-byte windows of a clean ROM's code segment that
are identical to the retail code segment must be explained by compiled
decomp output (the object/binary files our build produced from source).

    flagged = check(retail_code, clean_code, compiled=[bytes, ...], cfg=None)
    -> [(clean offset start, end)] merged byte ranges that are retail-only

Padding-like windows (fewer than min_nonzero nonzero bytes) are ignored.
"""
import numpy as np

from cleanroom.originality.config import CONFIG

P = np.uint64(0x100000001B3)


def window_hashes(buf, window=256, min_nonzero=32):
    a = np.frombuffer(bytes(buf), np.uint8)
    n = len(a) - window + 1
    if n <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.full(n, 0xCBF29CE484222325, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(window):
            h = (h ^ a[k:k + n].astype(np.uint64)) * P
    nz = np.concatenate([[0], np.cumsum(a != 0)])
    low = (nz[window:window + n] - nz[:n]) < min_nonzero
    return h, low


def check(retail_code, clean_code, compiled=(), cfg=None):
    cfg = cfg or CONFIG["code"]
    w = cfg["window"]
    rh, _ = window_hashes(retail_code, w, cfg["min_nonzero"])
    ch, low = window_hashes(clean_code, w, cfg["min_nonzero"])
    retail = np.unique(rh)
    allowed = np.unique(np.concatenate([window_hashes(b, w, 0)[0] for b in compiled] or [np.zeros(0, np.uint64)]))
    bad = np.isin(ch, retail) & ~np.isin(ch, allowed) & ~low
    idx = np.nonzero(bad)[0]
    ranges = []
    for i in idx:
        if ranges and i <= ranges[-1][1]:
            ranges[-1][1] = int(i) + w
        else:
            ranges.append([int(i), int(i) + w])
    return [tuple(r) for r in ranges]
