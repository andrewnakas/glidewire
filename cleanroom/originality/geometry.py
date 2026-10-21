"""Geometry originality: does a clean mesh occupy the same space as retail?

Both meshes are surface-sampled into voxels on one shared grid (voxel size
from config, default = largest bbox extent / voxel_div). Measures:

  voxel_iou       tolerant IoU: a voxel matches if the other mesh has one
                  within `tolerance` voxels; (matched_clean + matched_retail)
                  / (|clean| + |retail|). Voxels within anchor_radius of a
                  gameplay anchor (spawns, warps, triggers, path nodes) are
                  removed from both sides first: those may coincide.
  silhouettes     IoU of the occupied cells projected along X, Y and Z.

Fail: voxel_iou > voxel_iou_max, or mean silhouette > silhouette_max
together with voxel_iou > silhouette_voxel_min (a same-footprint level with
new shapes passes; a same-shape level does not).

    load_mesh(path) -> (verts (N,3) float, tris (M,3) int)   .obj / .npz / .json
    compare(retail, clean, anchors=None, cfg=None) -> dict
"""
import json

import numpy as np

from cleanroom.originality.config import CONFIG


def load_mesh(path):
    if path.endswith(".npz"):
        z = np.load(path)
        return z["verts"].astype(np.float64), z["tris"].astype(np.int64)
    if path.endswith(".json"):
        d = json.load(open(path))
        return np.asarray(d["verts"], np.float64), np.asarray(d["tris"], np.int64)
    vs, ts = [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            vs.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            idx = [int(x.split("/")[0]) for x in p[1:]]
            idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
            ts += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(vs, np.float64).reshape(-1, 3), np.asarray(ts, np.int64).reshape(-1, 3)


def sample_surface(verts, tris, step):
    """Points on every triangle, spaced <= step (barycentric grid)."""
    v = np.asarray(verts, np.float64)
    t = np.asarray(tris, np.int64)
    if not len(t):
        return np.zeros((0, 3))
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    edge = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                   np.linalg.norm(a - c, axis=1)], axis=0)
    k = np.minimum(np.ceil(edge / step).astype(int) + 1, 2048)
    out = [a, b, c]
    for kk in np.unique(k):
        sel = k == kk
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        keep = i + j <= kk
        u, w = (i[keep] / kk)[None, :, None], (j[keep] / kk)[None, :, None]
        A, B, C = a[sel][:, None], b[sel][:, None], c[sel][:, None]
        out.append((A + u * (B - A) + w * (C - A)).reshape(-1, 3))
    return np.concatenate(out)


def voxels(verts, tris, size, origin):
    p = sample_surface(verts, tris, size * 0.5)
    q = np.floor((p - origin) / size).astype(np.int64)
    return np.unique(q, axis=0)


def _keys(q):
    q = q + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def _matched(a, b, tol):
    """Rows of voxel set a with a voxel of b within tol (Chebyshev)."""
    kb = np.unique(_keys(b))
    hit = np.zeros(len(a), bool)
    r = range(-tol, tol + 1)
    for dx in r:
        for dy in r:
            for dz in r:
                ka = _keys(a + np.array([dx, dy, dz]))
                hit |= np.isin(ka, kb)
    return hit


def _drop_anchors(q, anchors, origin, size, radius):
    if anchors is None or not len(anchors) or not len(q):
        return q
    ac = (np.asarray(anchors, np.float64).reshape(-1, 3) - origin) / size
    keep = np.ones(len(q), bool)
    centers = q + 0.5
    for p in ac:
        keep &= np.max(np.abs(centers - p), axis=1) > radius
    return q[keep]


def silhouettes(q):
    return [set(map(tuple, np.unique(np.delete(q, ax, axis=1), axis=0))) for ax in range(3)]


def compare(retail, clean, anchors=None, cfg=None):
    cfg = cfg or CONFIG["geometry"]
    (rv, rt), (cv, ct) = retail, clean
    allv = np.concatenate([np.asarray(rv, float).reshape(-1, 3), np.asarray(cv, float).reshape(-1, 3)])
    origin = allv.min(axis=0)
    size = cfg.get("voxel") or max(float((allv.max(axis=0) - origin).max()), 1e-6) / cfg["voxel_div"]
    R = _drop_anchors(voxels(rv, rt, size, origin), anchors, origin, size, cfg["anchor_radius"])
    C = _drop_anchors(voxels(cv, ct, size, origin), anchors, origin, size, cfg["anchor_radius"])
    if not len(R) or not len(C):
        return {"voxel_iou": 0.0, "silhouette": [0.0] * 3, "voxel": size, "fail": False, "why": []}
    tol = cfg["tolerance"]
    iou = (_matched(C, R, tol).sum() + _matched(R, C, tol).sum()) / (len(C) + len(R))
    sil = [len(a & b) / max(1, len(a | b)) for a, b in zip(silhouettes(R), silhouettes(C))]
    why = []
    if iou > cfg["voxel_iou_max"]:
        why.append(f"voxel IoU {iou:.2f} > {cfg['voxel_iou_max']}")
    if np.mean(sil) > cfg["silhouette_max"] and iou > cfg["silhouette_voxel_min"]:
        why.append(f"silhouettes {np.mean(sil):.2f} > {cfg['silhouette_max']} with voxel IoU {iou:.2f}")
    return {"voxel_iou": round(float(iou), 4), "silhouette": [round(s, 4) for s in sil],
            "voxel": size, "score": round(float(iou) / cfg["voxel_iou_max"], 4),
            "fail": bool(why), "why": why}
