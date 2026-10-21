"""CLEAN ROOM: level kit. New terrain and architecture from parameters and a
seed; the only retail inputs are anchors (cleanroom.geo.anchors) and
surface-type ids.

    Mesh(verts, tris, surface, uv, color)    float32 (N,3), int32 (M,3), uint8 (M,), (N,2), (N,4) uint8
    heightfield(size_x, size_z, cell, seed, amp, anchors=...)   value-noise terrain, flattened at anchors
    box(lo, hi) / ramp(origin, width, length, height, axis)      modular pieces
    spline_track(points, width, bank=None, max_bank=25)          Catmull-Rom ribbon, banked by curvature
    merge(*meshes) / collision(mesh) -> (verts, tris, surface)   welded collision triangles

Y is up; coordinates are world units (N64 games: s16).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    verts: np.ndarray
    tris: np.ndarray
    surface: np.ndarray
    uv: np.ndarray = None
    color: np.ndarray = None

    def __post_init__(self):
        self.verts = np.asarray(self.verts, np.float32).reshape(-1, 3)
        self.tris = np.asarray(self.tris, np.int32).reshape(-1, 3)
        self.surface = np.broadcast_to(np.asarray(self.surface, np.uint8), (len(self.tris),)).copy()
        if self.uv is None:
            self.uv = self.verts[:, [0, 2]] / 64.0
        if self.color is None:
            self.color = np.full((len(self.verts), 4), 255, np.uint8)

    def bounds(self):
        return self.verts.min(axis=0), self.verts.max(axis=0)


def merge(*meshes):
    off, parts = 0, []
    for m in meshes:
        parts.append((m, off))
        off += len(m.verts)
    return Mesh(np.concatenate([m.verts for m, _ in parts]),
                np.concatenate([m.tris + o for m, o in parts]),
                np.concatenate([m.surface for m, _ in parts]),
                np.concatenate([m.uv for m, _ in parts]),
                np.concatenate([m.color for m, _ in parts]))


def _grid_tris(nx, nz):
    i = np.arange(nx - 1)[None, :] + nx * np.arange(nz - 1)[:, None]
    i = i.ravel()
    return np.concatenate([np.stack([i, i + nx, i + 1], 1), np.stack([i + 1, i + nx, i + nx + 1], 1)])


def value_noise(x, z, seed, octaves=4, scale=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x, dtype=np.float64)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        f = scale * 2 ** o
        g = rng.uniform(-1, 1, (int(f) + 3, int(f) + 3))
        u, v = x * f, z * f
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i, v - j
        fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
        a, b = g[i, j], g[i + 1, j]
        c, d = g[i, j + 1], g[i + 1, j + 1]
        out += amp * ((a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv)
        tot += amp
        amp *= 0.5
    return out / tot


def heightfield(size_x, size_z, cell, seed=0, amp=200.0, octaves=4, origin=(0, 0, 0), anchors=None,
                pad=2.0, surface=0, features=2.0):
    """Terrain over [origin, origin + size]; anchors (N,3) get a flat pad of
    `pad` cells at their own height, blended smoothly into the noise."""
    nx, nz = int(round(size_x / cell)) + 1, int(round(size_z / cell)) + 1
    xs = origin[0] + np.arange(nx) * cell
    zs = origin[2] + np.arange(nz) * cell
    X, Z = np.meshgrid(xs, zs)
    u = (X - origin[0]) / max(size_x, 1e-6)
    v = (Z - origin[2]) / max(size_z, 1e-6)
    Y = origin[1] + amp * value_noise(u, v, seed, octaves, features)
    if anchors is not None and len(anchors):
        for ax, ay, az in np.asarray(anchors, np.float64).reshape(-1, 3):
            d = np.hypot(X - ax, Z - az) / (pad * cell)
            w = np.clip(1.5 - d, 0, 1)
            w = w * w * (3 - 2 * w)
            Y = Y * (1 - w) + ay * w
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    return Mesh(verts, _grid_tris(nx, nz), surface)


def box(lo, hi, surface=0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    v = lo + c * (hi - lo)
    q = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = [t for a, b, c_, d in q for t in ((a, b, c_), (a, c_, d))]
    return Mesh(v, tris, surface)


def ramp(origin, width, length, height, axis="z", surface=0):
    """Wedge rising from 0 to `height` over `length` along +axis."""
    o = np.asarray(origin, float)
    v = np.array([[0, 0, 0], [width, 0, 0], [0, 0, length], [width, 0, length],
                  [0, height, length], [width, height, length]], float)
    if axis == "x":
        v = v[:, [2, 1, 0]]
    tris = [(0, 4, 1), (1, 4, 5), (0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 5, 4), (0, 2, 4), (1, 5, 3)]
    return Mesh(o + v, tris, surface)


def _catmull(p, samples):
    p = np.asarray(p, float)
    loop = len(p) > 2 and np.allclose(p[0], p[-1])
    if loop:
        p = p[:-1]
    n = len(p)
    pts = []
    segs = n if loop else n - 1
    for i in range(segs):
        p0 = p[(i - 1) % n] if loop or i > 0 else p[0]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if loop or i + 2 < n else p2
        t = np.linspace(0, 1, samples, endpoint=False)[:, None]
        pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts = np.concatenate(pts)
    return np.vstack([pts, pts[:1]]) if loop else np.vstack([pts, p[-1:]])


def spline_track(points, width, bank=None, max_bank=25.0, samples=8, surface=0, thickness=0.0):
    """Ribbon along a Catmull-Rom spline through `points` (closed if the last
    point equals the first). Bank (degrees per sample) defaults to curvature
    based banking, clamped to max_bank, rolling into the turn."""
    c = _catmull(points, samples)
    tan = np.gradient(c, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(tan, up)
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    if bank is None:
        turn = np.gradient(np.unwrap(np.arctan2(tan[:, 2], tan[:, 0])))
        bank = np.clip(np.degrees(turn) * 6.0, -max_bank, max_bank)
    b = np.radians(np.broadcast_to(np.asarray(bank, float), (len(c),)))
    n0 = np.cross(side, tan)                       # normal perpendicular to the tangent
    nup = np.cos(b)[:, None] * n0 + np.sin(b)[:, None] * side
    side_b = np.cross(tan, nup)
    side_b /= np.linalg.norm(side_b, axis=1, keepdims=True) + 1e-9
    left, right = c - side_b * width / 2, c + side_b * width / 2
    verts = np.empty((2 * len(c), 3))
    verts[0::2], verts[1::2] = left, right
    n = len(c)
    i = 2 * np.arange(n - 1)
    tris = np.concatenate([np.stack([i, i + 1, i + 2], 1), np.stack([i + 1, i + 3, i + 2], 1)])
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    uv = np.zeros((2 * n, 2), np.float32)
    uv[0::2, 0], uv[1::2, 0] = 0, 1
    uv[:, 1] = np.repeat(dist / max(width, 1e-6), 2)
    return Mesh(verts, tris, surface, uv)


def collision(mesh, weld=1e-3):
    """Welded vertex list + triangles + surface type per triangle; drops
    degenerate triangles."""
    q = np.round(mesh.verts / weld).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    t = inv[mesh.tris]
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return (uniq * weld).astype(np.float32), t[ok].astype(np.int32), mesh.surface[ok]
