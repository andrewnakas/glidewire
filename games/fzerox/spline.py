"""The game's course spline in Python (structure of Course_SplineCalculateTensions / Course_SplineGetPosition in
the decomp's src/game/course.c), used to validate generated courses before they reach the ROM.

    tensions(P) -> (alpha1, alpha2, ok) per segment; the game rejects a course if any alpha is outside [0, 2]
    position(P, T, i, t) -> point on segment i (cp i -> cp i+1) at t in [0, 1]
    lengths(P, T) -> per-segment arc lengths
"""
import numpy as np


def _alpha(prev, cur, nxt, nn):
    d1 = cur - nxt
    s = prev - nn + d1
    temp = (s @ s) / 16.0
    sp38 = (d1 @ s) * 0.75
    sp90 = sp38 * sp38
    sp88 = (d1 @ d1) * 9.0
    out = []
    for a, b in ((nxt, prev), (nn, cur)):
        v = a - b
        fv = temp - v @ v
        t2 = sp90 - fv * sp88
        if t2 >= 0:
            t2 = np.sqrt(t2)
            fv *= 2.0
            if fv < -0.1:
                out.append((sp38 - t2) / fv)
            elif fv > 0.1:
                out.append((sp38 + t2) / fv)
            else:
                out.append(-1.0)
        else:
            out.append(-1.0)
    return out


def tensions(P):
    P = np.asarray(P, float)
    n = len(P)
    a1, a2 = np.zeros(n), np.zeros(n)
    for i in range(n):
        a1[i], a2[i] = _alpha(P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n])
    ok = (a1 >= 0) & (a1 <= 2) & (a2 >= 0) & (a2 <= 2)
    return a1, a2, ok


def position(P, T, i, t):
    n = len(P)
    tension = T[i]
    t2, t3 = t * t, t * t * t
    w0 = (2 * t2 - t - t3) * tension
    w1 = (2 - tension) * t3 + (tension - 3) * t2 + 1
    w2 = (tension - 2) * t3 + (3 - 2 * tension) * t2 + tension * t
    w3 = (t3 - t2) * tension
    return (np.multiply.outer(w0, P[i - 1]) + np.multiply.outer(w1, P[i]) + np.multiply.outer(w2, P[(i + 1) % n])
            + np.multiply.outer(w3, P[(i + 2) % n]))


def curve(P, samples=20):
    """(points (n*samples, 3), segment index per point, tension per segment, ok)."""
    P = np.asarray(P, float)
    a1, a2, ok = tensions(P)
    T = (a1 + a2) * 0.5
    t = np.arange(samples) / samples
    pts = np.concatenate([position(P, T, i, t) for i in range(len(P))])
    seg = np.repeat(np.arange(len(P)), samples)
    return pts, seg, T, ok


def lengths(P, samples=20):
    pts, seg, T, ok = curve(P, samples)
    nxt = np.roll(pts, -1, axis=0)
    d = np.linalg.norm(nxt - pts, axis=1)
    return np.bincount(seg, d, minlength=len(P))
