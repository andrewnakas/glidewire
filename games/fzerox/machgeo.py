"""New 3D shapes for every real mesh in the machine / podium / course-object asset files.

Files: machine_global_gfx (30 racing machines x 5 LODs), machine_custom_gfx (custom-machine
fronts/rears/wings + the "super" machines + shadows), machine_models, podium_gfx,
course_track_gfx (gates, signs, buildings, mines, jump plates ...).

Kept facts (structure only): vertex count per array, the triangle index lists and which
array/slot each display list loads (from the decomp: asset-side gsSP* lists plus the code
that draws machines, ovl_i9/machine_draw.c and game/racer.c), per drawn object a bounding box
rounded to 16 units, per connected piece a UV box rounded to 32, a mean colour rounded to
32 levels and alpha to 4 levels, whether the array is lit (rgba = normal) or coloured, and
three relational bits read off retail positions: which vertices coincide (the weld that
makes split-normal duplicates one surface point), which vertex is the left/right mirror of
which, and each piece's winding sign (so back-face culling still shows the outside).

Topology: a "group" is one thing the game draws as a unit: one machine_draw.c function
(one machine LOD / one custom part LOD), one racer.c shadow card, or one root display list of
an asset file. Triangles join (array, index) nodes (no node is shared between groups).

New shapes. Each connected piece gets a direction per point from the spectral drawing of its
own triangle graph (low Laplacian eigenvectors; the mirror-antisymmetric one becomes x, then
symmetrised with the mirror pairs and lightly smoothed). The direction is cast as a ray onto a
body of our own design and the outermost hit is the new point:
  machines   six hull styles (arrow, dart, outrigger, delta, hammerhead, triple-boom), plus
             pods, fins, canopy and wing plates for the extra pieces; front / rear custom
             parts use the nose / tail half of the hull, wing parts the plates and fins;
  objects    podium/course pieces become twisted, profiled diamond prisms (taper, bulge,
             waist, step, flare) or bent panels with a new outline (Tutte layout).
Everything is then fitted to the rounded box. Thin axes (< 10 % of the largest extent: fins,
plates, signs) get a slab of 18 % of it and the piece is canted / bent across it.
Variant selection: per machine (all LODs together) the variants (style x mirror x jitter x
rake) are tried in a fixed order and the first whose every LOD scores below PASS_IOU on the
originality gate's voxel test is kept; retail is only the veto, never a target.
UVs are a planar projection of the new shape onto the rounded UV box; lit arrays get the
new vertex normals, coloured arrays get the rounded mean colour shaded by a fixed light.

Exemption: a group made only of flat 4-vertex rectangles (shadow cards, billboards) is a
layout primitive; it is rebuilt from its box rounded to 4 units and the UV corners, and is
not exported to the geometry gate.

No code reads asset vertex positions (checked: no .v.ob / pointer reads of these arrays in
src/ or the GLOBAL_ASM Racer_Draw/Course_GadgetsDraw; booster flames come from code tables),
so there are no anchors to keep. The 18 course_track_gfx DLs D_80140F0..D_8014640 are index
lists over the runtime course vertex buffer and use none of these arrays.

    write(dirty_tree, clean_tree)            rewrite Vtx initialisers in clean_tree in place
    export_gate(dirty_tree, clean_tree, d)   per-group meshes for cleanroom.originality
    python -m games.fzerox.machgeo check <dirty> <clean>
    python -m games.fzerox.machgeo gate <dirty> <clean> <gate_dir>
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

FILES = ["machine_global_gfx", "machine_custom_gfx", "machine_models", "podium_gfx", "course_track_gfx"]
ASSET_DIR = "src/assets/us/rev0"
CODE_FILES = ["src/overlays/ovl_i9/machine_draw.c", "src/game/racer.c"]

VTX_LINE = re.compile(r"^(\s*)\{\{\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\},\s*(-?\d+),\s*\{\s*(-?\d+),\s*(-?\d+)\s*\},"
                      r"\s*\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\}\}\},\s*$")
ARRAY_RE = re.compile(r"^Vtx (\w+)\[\] = \{\n(.*?)^\};", re.S | re.M)
GFX_RE = re.compile(r"^Gfx (\w+)\[\] = \{\n(.*?)^\};", re.S | re.M)
FUNC_RE = re.compile(r"^Gfx\* (\w+)\(Gfx\* gfx[^)]*\) \{\n(.*?)^\}", re.S | re.M)


def _seed(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ parsing
def read_arrays(path):
    """{sym: [(x, y, z, flag, s, t, r, g, b, a), ...]} in file order."""
    text = open(path, encoding="utf8").read()
    out = {}
    for m in ARRAY_RE.finditer(text):
        rows = []
        for line in m.group(2).splitlines():
            v = VTX_LINE.match(line)
            if v:
                rows.append(tuple(int(x) for x in v.groups()[1:]))
        out[m.group(1)] = rows
    return out


def _calls(text, prefix):
    out = []
    for m in re.finditer(r"\b(" + prefix + r"\w+)\(", text):
        i, depth = m.end(), 1
        while depth:
            depth += (text[i] == "(") - (text[i] == ")")
            i += 1
        args = [a.strip() for a in re.split(r",(?![^(]*\))", text[m.end():i - 1])]
        if args and args[0] == "gfx++":
            args = args[1:]
        out.append((m.group(1)[2:] if prefix == "gs" else m.group(1)[1:], args))
    return out


def _int(s):
    return int(s, 0)


class Topo:
    """Arrays, groups and triangles over (sym, index) nodes."""

    def __init__(self, tree):
        self.tree = tree
        self.arrays = {}      # sym -> rows
        self.file_of = {}     # sym -> file
        self.dls = {}         # dl sym -> [(call, args)]
        for f in FILES:
            arr = read_arrays(os.path.join(tree, ASSET_DIR, f, f + ".c"))
            for s, rows in arr.items():
                self.arrays[s] = rows
                self.file_of[s] = f
            text = open(os.path.join(tree, ASSET_DIR, f, f + ".c"), encoding="utf8").read()
            for m in GFX_RE.finditer(text):
                self.dls[m.group(1)] = (f, _calls(m.group(2), "gs"))
        self.group_tris = {}  # group -> [(n, n, n)]
        self.group_order = []
        self.missing = 0
        self._code()
        self._assets()

    def _add(self, group, tri):
        if group not in self.group_tris:
            self.group_tris[group] = []
            self.group_order.append(group)
        self.group_tris[group].append(tri)

    def _run(self, group, calls, slots, depth=0):
        for name, args in calls:
            if name == "SPVertex":
                a0 = args[0]
                sym = a0.lstrip("&").split("[")[0].strip()
                if sym not in self.arrays:
                    continue
                base = _int(a0.split("[")[1].rstrip("]")) if "[" in a0 else 0
                n, v0 = _int(args[1]), _int(args[2])
                for k in range(n):
                    slots[v0 + k] = (sym, base + k)
            elif name in ("SP1Triangle", "SP2Triangles"):
                idx = [_int(a) for a in args]
                for g in [idx[0:3]] + ([idx[4:7]] if name == "SP2Triangles" else []):
                    if all(v in slots for v in g):
                        self._add(group, tuple(slots[v] for v in g))
                    else:
                        self.missing += 1
            elif name == "SPDisplayList" and depth < 8:
                sym = args[0].strip()
                if sym in self.dls:
                    self.called.add(sym)
                    self._run(group, self.dls[sym][1], slots, depth + 1)

    def _code(self):
        self.called = set()
        md = open(os.path.join(self.tree, CODE_FILES[0]), encoding="utf8").read()
        for m in FUNC_RE.finditer(md):
            calls = _calls(m.group(2), "g")
            if any(c[0] == "SPVertex" for c in calls):
                self._run("machine_draw:" + m.group(1), calls, {})
        # racer.c: shadow cards (D_800CDBA4[] table and the crash shadow), triangles 0,1,2 / 0,3,1
        rc = open(os.path.join(self.tree, CODE_FILES[1]), encoding="utf8").read()
        tab = re.search(r"Vtx\* D_800CDBA4\[\] = \{(.*?)\};", rc, re.S)
        syms = re.findall(r"\w+", tab.group(1)) if tab else []
        for s in syms + ["D_7048868"]:
            if s in self.arrays:
                self._run("racer_shadow:" + s, [("SPVertex", [s, "4", "0"]),
                                                ("SP2Triangles", ["0", "1", "2", "0", "0", "3", "1", "0"])], {})
        self.code_called = set(self.called)

    def _assets(self):
        self.called = set()
        for dl, (f, calls) in self.dls.items():
            for name, args in calls:
                if name == "SPDisplayList" and args[0].strip() in self.dls:
                    self.called.add(args[0].strip())
        for dl, (f, calls) in self.dls.items():
            if dl in self.called:
                continue
            self._run(f + ":" + dl, calls, {})

    # ------------------------------------------------------------ derived
    def owners(self):
        """node -> group (first group in order that uses it)."""
        own = {}
        for g in self.group_order:
            for t in self.group_tris[g]:
                for n in t:
                    own.setdefault(n, g)
        return own


# ------------------------------------------------------------------ per-group structure
def _rnd(x, q):
    return np.round(np.asarray(x, np.float64) / q) * q


def _comps(nu, tris):
    par = list(range(nu))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b, c in tris:
        par[f(a)] = f(b)
        par[f(b)] = f(c)
    cc = {}
    for i in range(nu):
        cc.setdefault(f(i), []).append(i)
    return sorted(cc.values(), key=lambda c: (-len(c), c[0]))


class Group:
    """One drawn object: nodes, welded points (nodes at one retail position share a point),
    triangles over points, connected pieces. Retail positions are only read for the
    rounded boxes, the weld and the card test."""

    def __init__(self, topo, name):
        self.name = name
        self.tris_n = topo.group_tris[name]
        self.nodes = sorted({n for t in self.tris_n for n in t}, key=lambda n: (n[0], n[1]))
        self.nidx = {n: i for i, n in enumerate(self.nodes)}
        rows = [topo.arrays[s][i] for s, i in self.nodes]
        self.rows = rows
        P = np.array([r[:3] for r in rows], float)
        self.retail = P
        key = {}
        self.pt = np.array([key.setdefault(tuple(p), len(key)) for p in P.tolist()])
        self.npt = len(key)
        tp = []
        for t in self.tris_n:
            q = tuple(self.pt[self.nidx[n]] for n in t)
            if len(set(q)) == 3:
                tp.append(q)
        self.tris_p = tp
        self.comps = _comps(self.npt, tp)
        Pp = np.zeros((self.npt, 3))
        Pp[self.pt] = P
        self.box = (_rnd(P.min(0), 16), _rnd(P.max(0), 16))
        self.cbox = [(_rnd(Pp[c].min(0), 16), _rnd(Pp[c].max(0), 16)) for c in self.comps]
        self.cards = [self._is_rect(Pp[c]) for c in self.comps]
        files = [topo.file_of[n[0]] for n in self.nodes]
        self.file = max(sorted(set(files)), key=files.count)
        # orientation bit per piece: sign of the signed volume about the piece centre
        self.orient = []
        for c in self.comps:
            cs = set(c)
            m = Pp[c].mean(0)
            v = sum(np.dot(Pp[a] - m, np.cross(Pp[b] - m, Pp[cc] - m)) for a, b, cc in tp if a in cs)
            self.orient.append(1 if v >= 0 else -1)
        self.retail_pts = Pp
        # mirror partner of each point (x -> -x), itself on the centre plane, -1 if none
        # (mutual nearest mirror image within 3 units about the group's mid-x plane)
        cx = (Pp[:, 0].min() + Pp[:, 0].max()) / 2
        M = Pp.copy()
        M[:, 0] = 2 * cx - M[:, 0]
        D = np.linalg.norm(Pp[None, :, :] - M[:, None, :], axis=2)
        nn = D.argmin(1)
        self.mirror = [int(nn[i]) if D[i, nn[i]] <= 3 and nn[nn[i]] == i else -1 for i in range(len(Pp))]

    @staticmethod
    def _is_rect(p):
        if len(p) != 4:
            return False
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-6 or abs(np.dot(p[3] - p[0], n / np.linalg.norm(n))) > 1.0:
            return False
        for i, j in ((0, 1), (0, 2), (0, 3)):
            k, l = [x for x in range(4) if x not in (i, j)]
            if np.abs(p[i] + p[j] - p[k] - p[l]).max() <= 1.5 and \
                    abs(np.linalg.norm(p[i] - p[j]) - np.linalg.norm(p[k] - p[l])) <= 2.0:
                return True
        return False

    @property
    def is_card(self):
        return all(self.cards)

    @property
    def kind(self):
        g = self.name.split(":", 1)[1]
        if self.name.startswith(("machine_draw", "machine_models", "machine_global_gfx")):
            if "Wing" in g:
                return "wing"
            if "Front" in g:
                return "front"
            if "Rear" in g:
                return "rear"
            return "machine"
        return "object"

    @property
    def style_key(self):
        g = self.name.split(":", 1)[1]
        return re.sub(r"Lod(\d+|Furthest|Unused)$", "", g)


# ------------------------------------------------------------------ intrinsic coordinates
def _eig(n, tris):
    """Low Laplacian eigenvectors of a piece's own triangle graph (columns 1..3)."""
    A = np.zeros((n, n))
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            A[x, y] = A[y, x] = 1
    deg = A.sum(1)
    deg[deg == 0] = 1
    L = np.eye(n) - A / np.sqrt(deg[:, None] * deg[None, :])
    _, vec = np.linalg.eigh(L)
    e = np.zeros((n, 3))
    k = min(3, n - 1)
    e[:, :k] = vec[:, 1:1 + k]
    for j in range(3):                      # fix eigenvector sign deterministically
        if np.abs(e[:, j]).max() > 0 and e[np.argmax(np.abs(e[:, j])), j] < 0:
            e[:, j] *= -1
    return e, A


def _rank(x):
    n = len(x)
    if n == 1:
        return np.array([0.5])
    return np.argsort(np.argsort(x + 1e-9 * np.arange(n), kind="stable"), kind="stable") / (n - 1)


def _symmetrize(U, mir):
    F = np.array([-1.0, 1.0, 1.0])
    for i, m in enumerate(mir):
        if m < 0 or m < i:
            continue
        if m == i:
            U[i, 0] = 0
        else:
            U[i] = (U[i] + F * U[m]) / 2
            U[m] = F * U[i]
    return U


def _dirs(n, tris, mir, seed):
    """Unit directions per point (design axes x, y, z = across, up, length) from the
    spectral drawing of the piece's own graph, made mirror-symmetric with the kept pairs."""
    rng = np.random.default_rng(seed)
    if n < 4:
        base = np.array([[0, 1, -1], [-1, -0.3, 0.6], [1, -0.3, 0.6]], float)[:n]
        return base / np.linalg.norm(base, axis=1, keepdims=True), np.eye(n)
    e, A = _eig(n, tris)
    sd = e.std(0)
    sd[sd < 1e-9] = 1
    e = e / sd
    xcol = 2
    pairs = [(i, m) for i, m in enumerate(mir) if m >= 0]
    if pairs:
        sc = []
        for c in range(3):
            num = sum((e[i, c] + e[m, c]) ** 2 for i, m in pairs)
            den = sum(e[i, c] ** 2 + e[m, c] ** 2 for i, m in pairs) + 1e-9
            sc.append(num / den)
        xcol = int(np.argmin(sc))
    rest = [c for c in range(3) if c != xcol]
    U = np.stack([e[:, xcol], e[:, rest[1]], e[:, rest[0]]], 1)
    U -= U.mean(0)
    U = _symmetrize(U, mir)
    for _ in range(3):
        nb = A @ U / np.maximum(A.sum(1, keepdims=True), 1)
        U = 0.65 * U + 0.35 * nb
        ln = np.linalg.norm(U, axis=1, keepdims=True)
        small = ln[:, 0] < 1e-6
        U[small] = rng.normal(size=(small.sum(), 3))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        U = _symmetrize(U, mir)
    return U, A


# ------------------------------------------------------------------ bodies (design space)
def _raycast(inside, c, d):
    """Outermost surface point of the body along c + t d (t in (0, 3])."""
    ts = np.linspace(3.0, 0.0, 241)
    P = c[None, None, :] + ts[:, None, None] * d[None, :, :]
    ins = inside(P)
    first = np.argmax(ins, axis=0)
    hit = ins.any(axis=0)
    t_in = np.where(hit, ts[first], 0.0)
    t_out = np.where(hit & (first > 0), ts[np.maximum(first - 1, 0)], t_in)
    for _ in range(12):
        mid = (t_in + t_out) / 2
        ok = inside(c[None, :] + mid[:, None] * d)
        t_in = np.where(ok, mid, t_in)
        t_out = np.where(ok, t_out, mid)
    return c[None, :] + t_in[:, None] * d


def _ell(P, c, r):
    return (((P - c) / r) ** 2).sum(-1) <= 1


HULLS = ["wedge", "dart", "twinpod", "delta", "hammer", "boom"]


def _body(style, j):
    """inside(P) for a hull style; design space x in [-1, 1], y in [0, 1], z in [0, 1] tail -> nose."""
    def wedge(P):      # "arrow": slim tapered spine, high rear spoiler on twin struts
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zc = np.clip(z, 0, 1)
        W = 0.26 - (0.18 + 0.03 * j[0]) * zc
        top = 0.22 + 0.33 * (1 - zc) ** 1.2
        ym, hh = (top + 0.12) / 2, (top - 0.12) / 2
        spine = (z >= 0) & (z <= 1) & (np.abs(x / W) ** 4 + np.abs((y - ym) / hh) ** 2 <= 1)
        ax = np.abs(x)
        spoiler = (ax <= 1) & (np.abs(y - (0.9 - 0.06 * ax)) <= 0.035) & (z >= 0.0) & (z <= 0.14 + 0.04 * ax)
        strut = (np.abs(ax - 0.3) <= 0.03) & (y >= 0.3) & (y <= 0.9) & (z >= 0.02) & (z <= 0.12)
        keel = (ax <= 0.03) & (y >= 0.0) & (y <= 0.15) & (z >= 0.25) & (z <= 0.75)
        return spine | spoiler | strut | keel

    def dart(P):       # slim fuselage, thin dihedral wings, tall tail fin
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zc = np.clip(z, 0, 1)
        W = 0.2 * np.sin(np.pi * (0.06 + 0.9 * zc)) ** 0.6 + 0.02
        hh = 0.17 * np.sin(np.pi * (0.04 + 0.92 * zc)) ** 0.5 + 0.02
        fus = (z >= 0) & (z <= 1) & ((x / W) ** 2 + ((y - 0.4) / hh) ** 2 <= 1)
        ax = np.abs(x)
        wing = (np.abs(y - (0.05 + 0.3 * ax)) <= 0.03) & (z >= 0.04) & (z <= 0.5 - (0.36 + 0.08 * j[0]) * ax) & (ax <= 1)
        fin = (ax <= 0.03) & (y >= 0.45) & (y <= 1.0) & (z >= 0) & (z <= 0.26 - 0.2 * (y - 0.45) / 0.55)
        return fus | wing | fin

    def twinpod(P):    # "outrigger": slim centre pod, two small side pods on a thin cross plate
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zc = np.clip(z, 0, 1)
        T = np.sin(np.pi * (0.03 + 0.94 * zc)) ** 0.5
        core = (z >= 0) & (z <= 1) & ((x / 0.2) ** 2 + ((y - 0.55) / 0.3) ** 2 <= T ** 2)
        T2 = np.sin(np.pi * np.clip(z / 0.6, 0, 1)) ** 0.5
        pods = np.zeros(x.shape, bool)
        for sx in (-1, 1):
            pods |= (z >= 0) & (z <= 0.6) & (((x - sx * 0.86) / 0.14) ** 2 + ((y - 0.2) / 0.2) ** 2 <= T2 ** 2)
        plate = (np.abs(x) <= 0.86) & (np.abs(y - (0.3 + 0.1 * j[0])) <= 0.025) & (z >= 0.2) & (z <= 0.42)
        return core | pods | plate

    def delta(P):      # flat swept delta with dihedral and a small canopy
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        W = 0.04 + 0.96 * np.clip(1 - z, 0, 1) ** (0.9 + 0.2 * j[0])
        ax = np.abs(x)
        base = 0.02 + 0.55 * ax
        th = 0.08 * np.clip(1 - ax / W, 0, 1)
        plate = (z >= 0) & (z <= 1) & (ax <= W) & (y >= base) & (y <= base + th)
        can = _ell(P, np.array([0, 0.1, 0.5]), np.array([0.13, 0.28, 0.22]))
        return plate | can

    def hammer(P):     # slim spine, wide flat canard across the nose, small rear pods
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zc = np.clip(z, 0, 1)
        ax = np.abs(x)
        spine = (z >= 0) & (z <= 1) & ((x / 0.16) ** 2 + ((y - 0.5) / (0.22 + 0.1 * (1 - zc))) ** 2 <= 1)
        canard = (ax <= 1) & (np.abs(y - (0.62 - 0.1 * ax)) <= 0.04) & (z >= 0.72 + 0.1 * ax) & (z <= 0.95)
        pods = np.zeros(x.shape, bool)
        for sx in (-1, 1):
            pods |= _ell(P, np.array([sx * 0.45, 0.18, 0.15]), np.array([0.12, 0.18, 0.15 + 0.03 * j[1]]))
        return spine | canard | pods

    def boom(P):       # three thin parallel booms under a high flat deck
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zc = np.clip(z, 0, 1)
        T = np.sin(np.pi * (0.02 + 0.96 * zc)) ** 0.4
        out = (np.abs(y - 0.85) <= 0.06) & (np.abs(x) <= 0.7) & (z >= 0.3) & (z <= 0.6 + 0.1 * j[0])
        for sx, r in ((-0.8, 0.12), (0.0, 0.18), (0.8, 0.12)):
            out |= (z >= 0) & (z <= 1) & (((x - sx) / r) ** 2 + ((y - 0.35) / 0.3) ** 2 <= T ** 2)
        return out

    f = {"wedge": wedge, "dart": dart, "twinpod": twinpod, "delta": delta, "hammer": hammer, "boom": boom}[style]
    c = {"wedge": (0, 0.3, 0.4), "dart": (0, 0.4, 0.45), "twinpod": (0, 0.55, 0.45), "delta": (0, 0.12, 0.5),
         "hammer": (0, 0.5, 0.5), "boom": (0, 0.4, 0.5)}[style]
    return f, np.array(c, float)


def _part_body(role, j):
    """Secondary machine pieces: (inside, centre) in the hull design space."""
    if role in ("podL", "podR"):
        sx = -1 if role == "podL" else 1
        c = np.array([sx * 0.86, 0.22, 0.25])
        return (lambda P: _ell(P, c, np.array([0.13, 0.13, 0.27 + 0.04 * j[1]]))), c
    if role == "fin":
        c = np.array([0, 0.75, 0.14])
        return (lambda P: (np.abs(P[..., 0]) <= 0.03) & (P[..., 1] >= 0.5) & (P[..., 1] <= 1) &
                (P[..., 2] >= 0) & (P[..., 2] <= 0.3 - 0.2 * (P[..., 1] - 0.5) / 0.5)), c
    if role == "canopy":
        c = np.array([0, 0.66, 0.6])
        return (lambda P: _ell(P, c, np.array([0.18, 0.14, 0.24]))), c
    sx = -1 if role == "wingL" else 1      # swept, dihedral plates
    c = np.array([sx * 0.72, 0.3, 0.2])
    return (lambda P: (sx * P[..., 0] >= 0.42) & (sx * P[..., 0] <= 1) &
            (np.abs(P[..., 1] - (0.2 + 0.2 * (sx * P[..., 0] - 0.42))) <= 0.035) &
            (P[..., 2] >= 0.02 + 0.3 * (sx * P[..., 0] - 0.42)) & (P[..., 2] <= 0.42 + 0.1 * (sx * P[..., 0] - 0.42) ** 2)), c


ROLES = ["podL", "podR", "fin", "canopy", "wingL", "wingR"]
DSTRETCH = np.array([1.0, 0.55, 0.6])


def _fit(p, lo, hi):
    """Scale/translate each axis of p onto [lo, hi] (keeps the rounded box)."""
    p = np.asarray(p, float)
    a, b = p.min(0), p.max(0)
    span = np.where(b - a < 1e-9, 1, b - a)
    q = (p - a) / span
    q = np.where((b - a)[None, :] < 1e-9, 0.5, q)
    return lo + q * (hi - lo)


def _signed_volume(p, tris, cs):
    m = p[cs].mean(0)
    sset = set(cs)
    return sum(np.dot(p[a] - m, np.cross(p[b] - m, p[c] - m)) for a, b, c in tris if a in sset)


def _orient_fix(p, tris, cs, want):
    """Mirror a piece in x about its centre if its signed volume disagrees with the kept bit."""
    v = _signed_volume(p, tris, cs)
    if (1 if v >= 0 else -1) != want and abs(v) > 1e-6:
        m = p[cs].mean(0)
        p[cs, 0] = 2 * m[0] - p[cs, 0]
    return p


N_BASE = 24
N_DIAG = 8
N_VARIANTS = 3 * N_BASE + N_DIAG   # base designs x (no shear, rake fwd, rake back) + diagonal spindles
PASS_IOU = 0.33      # selection target, below the gate's 0.40


THIN = 0.1


def _thin_axes(box):
    ext = box[1] - box[0]
    return [ax for ax in range(3) if ext[ax] < THIN * max(ext.max(), 1)]


def _thicken(lo, hi):
    """Thin axes (flat fins, plates) get a slab of 0.18 x the largest extent, centred on the
    old one: the new piece is canted out of the retail plane."""
    lo, hi = np.asarray(lo, float).copy(), np.asarray(hi, float).copy()
    mx = max((hi - lo).max(), 1)
    for ax in _thin_axes((lo, hi)):
        c = (lo[ax] + hi[ax]) / 2
        h = max(0.09 * mx, 8)
        lo[ax], hi[ax] = _rnd(c - h, 8), _rnd(c + h, 8)
    return lo, hi


def _roles(g, kind, variant):
    """Role per extra piece: pieces that are each other's mirror image become a left/right
    pair (pods, wing plates), self-symmetric pieces sit on the centre line (canopy, fin)."""
    comp_of = {}
    for ci, c in enumerate(g.comps):
        for v in c:
            comp_of[v] = ci
    roles, k_c, k_p = {}, variant % 2, (variant // 2) % 2
    start = 0 if kind == "wing" else 1
    for ci in range(start, len(g.comps)):
        if ci in roles:
            continue
        c = g.comps[ci]
        partners = [comp_of[g.mirror[v]] for v in c if g.mirror[v] >= 0]
        other = max(set(partners), key=partners.count) if partners else ci
        if other != ci and other not in roles and partners.count(other) >= len(c) / 2:
            pair = [("podL", "podR"), ("wingL", "wingR")][(k_p + (kind == "wing")) % 2]
            k_p += 1
            roles[ci], roles[other] = pair
        elif partners.count(ci) >= len(c) / 2 or kind in ("front", "wing"):
            roles[ci] = ["canopy", "fin"][k_c % 2]
            k_c += 1
        else:
            roles[ci] = ["podL", "podR"][ci % 2]
    return roles


DIAGS = [((0, 0, 0), (1, 1, 1)), ((1, 0, 0), (0, 1, 1)), ((0, 1, 0), (1, 0, 1)), ((0, 0, 1), (1, 1, 0))]


def _diagonal_group(g, d):
    """Last-resort design for tiny pieces: a twisted spindle along a diagonal of the box
    (objects: thin axes widened to 30 % of the largest extent)."""
    P = np.zeros((g.npt, 3))
    boxes = g.cbox if g.kind == "object" and d < 4 else [g.box] * len(g.comps)
    for ci, c in enumerate(g.comps):
        ix = {v: k for k, v in enumerate(c)}
        tr = [(ix[a], ix[b], ix[cc]) for a, b, cc in g.tris_p if a in ix]
        mir = [ix.get(g.mirror[v], -1) if g.mirror[v] >= 0 else -1 for v in c]
        U, _ = _dirs(len(c), tr, mir, _seed(g.name) + ci)
        lo, hi = np.asarray(boxes[ci][0], float).copy(), np.asarray(boxes[ci][1], float).copy()
        mx = max((hi - lo).max(), 1)
        for ax in range(3):
            if hi[ax] - lo[ax] < (0.2 if g.kind == "object" else THIN) * mx:
                cc_ = (lo[ax] + hi[ax]) / 2
                lo[ax], hi[ax] = _rnd(cc_ - 0.15 * mx, 8), _rnd(cc_ + 0.15 * mx, 8)
        A, B = np.array(DIAGS[(d + ci) % 4][0], float), np.array(DIAGS[(d + ci) % 4][1], float)
        ax_ = (B - A) / np.linalg.norm(B - A)
        e1 = np.cross(ax_, [0.0, 1.0, 0.0] if abs(ax_[1]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax_, e1)
        t = _rank(U[:, 2])
        ph = np.arctan2(U[:, 1], U[:, 0]) + 2.5 * t
        r = 0.22 * np.sin(np.pi * (0.1 + 0.8 * t))
        q = A[None] + t[:, None] * (B - A)[None] + r[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
        P[c] = _fit(q, lo, hi)
    for ci, c in enumerate(g.comps):
        P = _orient_fix(P, g.tris_p, c, g.orient[ci])
    return P


def shape_group(g, variant=0):
    """New point positions (npt, 3) for a group, inside its rounded box. The variant turns
    the style / mirror / rake choices; all variants are our own designs."""
    if variant >= 3 * N_BASE:
        return _diagonal_group(g, variant - 3 * N_BASE)
    shear = [0, 1, -1][variant // N_BASE]
    variant = variant % N_BASE
    rng = np.random.default_rng(_seed(g.style_key) + 104729 * (variant // (2 * len(HULLS))))
    j = rng.uniform(-1, 1, 4)
    P = np.zeros((g.npt, 3))
    lo, hi = g.box
    kind = g.kind
    local, lmir = {}, {}
    for ci, c in enumerate(g.comps):
        ix = {v: k for k, v in enumerate(c)}
        local[ci] = [(ix[a], ix[b], ix[cc]) for a, b, cc in g.tris_p if a in ix]
        lmir[ci] = [ix.get(g.mirror[v], -1) if g.mirror[v] >= 0 else -1 for v in c]
    if kind == "object":
        for ci, c in enumerate(g.comps):
            P[c] = _object_piece(len(c), local[ci], lmir[ci], g.cbox[ci], _seed(g.name) + ci, variant, shear)
    else:
        style = HULLS[(_seed(g.style_key) + variant) % len(HULLS)]
        flipz = (variant // len(HULLS)) % 2
        roles = _roles(g, kind, variant)
        for ci, c in enumerate(g.comps):
            U, _ = _dirs(len(c), local[ci], lmir[ci], _seed(g.name) + ci)
            if flipz:
                U[:, 2] *= -1
            if kind != "wing" and ci == 0:
                inside, c0 = _body(style, j)
                if kind == "front":
                    inside0 = inside
                    inside = lambda Q, f=inside0: f(Q) & (Q[..., 2] >= 0.45)
                    c0 = c0.copy()
                    c0[2] = max(c0[2], 0.6)
                elif kind == "rear":
                    inside0 = inside
                    inside = lambda Q, f=inside0: f(Q) & (Q[..., 2] <= 0.55)
                    c0 = c0.copy()
                    c0[2] = min(c0[2], 0.35)
            else:
                inside, c0 = _part_body(roles[ci], j)
            P[c] = _raycast(inside, c0, U * DSTRETCH)
        lo, hi = _thicken(lo, hi)
        if shear:                                   # rake: nose dips / lifts, upper parts swept
            P[:, 1] += shear * 0.22 * (P[:, 2] - 0.5)
            P[:, 2] -= shear * 0.12 * P[:, 1]
        for f in _thin_axes(g.box):
            a = int(np.argmax(np.ptp(P, 0)))
            P[:, f] = P[:, f] + 0.35 * (P[:, a] - P[:, a].min()) / max(np.ptp(P[:, a]), 1e-9)
        P = _fit(P, lo, hi)
    for ci, c in enumerate(g.comps):
        P = _orient_fix(P, g.tris_p, c, g.orient[ci])
    return P


OBJ_PROFILES = ["taper", "bulge", "waist", "step", "flare"]


def _boundary_cycle(nv, tris):
    cnt = {}
    for t in tris:
        for x, y in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (min(x, y), max(x, y))
            cnt[e] = cnt.get(e, 0) + 1
    adj = {}
    for (x, y), c in cnt.items():
        if c == 1:
            adj.setdefault(x, []).append(y)
            adj.setdefault(y, []).append(x)
    if not adj or any(len(v) != 2 for v in adj.values()):
        return None
    start = min(adj)
    cyc, prev, cur = [start], None, start
    while True:
        nxt = [y for y in adj[cur] if y != prev][0]
        if nxt == start:
            break
        cyc.append(nxt)
        prev, cur = cur, nxt
        if len(cyc) > nv:
            return None
    return cyc if len(cyc) == len(adj) else None


def _panel(n, tris, seed):
    """Flat piece: Tutte layout with the boundary on a new outline (unit square coords)."""
    cyc = _boundary_cycle(n, tris)
    shape = seed % 3
    if cyc is None or len(cyc) < 3:
        e = _eig(n, tris)[0] if n >= 4 else np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)[:n]
        return np.stack([_rank(e[:, 0]), _rank(e[:, 1])], 1)
    m = len(cyc)
    ang = 2 * np.pi * np.arange(m) / m + 0.3 * seed
    if shape == 0:      # lens
        out = np.stack([0.5 + 0.5 * np.cos(ang), 0.5 + 0.5 * np.sin(ang) * np.abs(np.sin(ang)) ** 0.0 * 0.6], 1)
    elif shape == 1:    # pennant (triangle-ish)
        out = np.stack([0.5 + 0.5 * np.cos(ang), 0.5 + 0.5 * np.sin(ang) * (0.5 + 0.5 * np.cos(ang))], 1)
    else:               # rounded diamond
        c, s = np.cos(ang), np.sin(ang)
        out = np.stack([0.5 + 0.5 * np.sign(c) * np.abs(c) ** 2, 0.5 + 0.5 * np.sign(s) * np.abs(s) ** 2], 1)
    pos = np.zeros((n, 2))
    fixed = np.zeros(n, bool)
    pos[cyc] = out
    fixed[cyc] = True
    nb = [set() for _ in range(n)]
    for t in tris:
        for x, y in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            nb[x].add(y)
            nb[y].add(x)
    free = [v for v in range(n) if not fixed[v]]
    if free:
        ix = {v: k for k, v in enumerate(free)}
        A = np.zeros((len(free), len(free)))
        B = np.zeros((len(free), 2))
        for v in free:
            A[ix[v], ix[v]] = max(len(nb[v]), 1)
            for w in nb[v]:
                if fixed[w]:
                    B[ix[v]] += pos[w]
                else:
                    A[ix[v], ix[w]] -= 1
            if not nb[v]:
                B[ix[v]] = [0.5, 0.5]
        pos[free] = np.linalg.lstsq(A, B, rcond=None)[0]
    return pos


def _object_piece(n, tris, mir, box, seed, variant=0, shear=0):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    ext = hi - lo
    rng = np.random.default_rng(seed + 7919 * variant)
    mx = max(ext.max(), 1)
    flat = [ax for ax in range(3) if ext[ax] < THIN * mx]
    if flat:                                 # flat panel -> new outline, bent out of its plane
        f = flat[0]
        a, b = [x for x in range(3) if x != f]
        if ext[b] > ext[a]:
            a, b = b, a
        uv = _panel(n, tris, seed + variant)
        p = np.zeros((n, 3))
        p[:, a], p[:, b] = uv[:, 0], uv[:, 1]
        p[:, f] = np.sin(np.pi * uv[:, 0]) * (0.6 + 0.4 * np.cos(2 * np.pi * uv[:, 1]))
        hi2, lo2 = hi.copy(), lo.copy()
        cf, h = (lo[f] + hi[f]) / 2, max(0.09 * mx, 8)
        lo2[f], hi2[f] = cf - h, cf + h
        return _fit(p, lo2, hi2)
    axis = 1 if ext[1] >= 0.5 * mx else int(np.argmax(ext))
    if (variant // 2) % 3:
        axis = int(np.argsort(-ext)[(variant // 2) % 3])
    a, b = [x for x in range(3) if x != axis]
    prof = OBJ_PROFILES[(seed + variant) % len(OBJ_PROFILES)]
    twist0 = np.pi / 4 + rng.uniform(-0.25, 0.25)
    twr = rng.uniform(0.4, 1.1)
    pe = rng.uniform(1.5, 2.4)

    def inside(P):
        s = P[..., 2]
        sc = np.clip(s, 0, 1)
        r = {"taper": 1 - 0.55 * sc, "bulge": 0.55 + 0.45 * np.sin(np.pi * sc), "waist": 1 - 0.5 * np.sin(np.pi * sc),
             "step": np.where(sc < 0.45, 1.0, 0.55), "flare": 0.45 + 0.55 * sc ** 1.5}[prof]
        tw = twist0 + twr * sc
        x = P[..., 0] * np.cos(tw) + P[..., 1] * np.sin(tw)
        y = -P[..., 0] * np.sin(tw) + P[..., 1] * np.cos(tw)
        return (s >= 0) & (s <= 1) & (np.abs(x) ** pe + np.abs(y) ** pe <= r ** pe)
    U, _ = _dirs(n, tris, mir, seed)
    if variant % 2:
        U[:, 2] *= -1
    q = _raycast(inside, np.array([0, 0, 0.5]), U * np.array([1, 1, 0.8]))
    q[:, 0] += shear * 0.35 * (q[:, 2] - 0.5)     # lean
    p = np.zeros((n, 3))
    p[:, a], p[:, b], p[:, axis] = q[:, 0], q[:, 1], q[:, 2]
    return _fit(p, lo, hi)


# ------------------------------------------------------------------ vertex rows
LIGHT = np.array([0.35, 0.8, 0.5]) / np.linalg.norm([0.35, 0.8, 0.5])


def _node_normals(g, Pn):
    nrm = np.zeros_like(Pn)
    for t in g.tris_n:
        a, b, c = (g.nidx[x] for x in t)
        nrm[[a, b, c]] += np.cross(Pn[b] - Pn[a], Pn[c] - Pn[a])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    bad = ln[:, 0] < 1e-9
    nrm[bad] = [0, 1, 0]
    ln[bad] = 1
    return nrm / ln


def _is_lit(rows):
    a = np.array([r[6:9] for r in rows], float)
    a = np.where(a > 127, a - 256, a)
    L = np.linalg.norm(a, axis=1)
    return np.mean((L > 100) & (L < 140)) > 0.5


def rows_for_group(g, lit_of, variant=0):
    """{node: new row} for every node of the group."""
    R = np.array(g.rows, float)
    if g.is_card:
        Pn = _rnd(g.retail, 4)
    else:
        Pp = shape_group(g, variant)
        Pn = Pp[g.pt]
    Pn = np.clip(np.round(Pn), -32768, 32767)
    nrm = _node_normals(g, Pn)
    comp_of = np.zeros(g.npt, int)
    for ci, c in enumerate(g.comps):
        comp_of[c] = ci
    out = {}
    # UV: per (array, piece) rounded UV box; planar projection of the new piece
    keys = {}
    for k, n in enumerate(g.nodes):
        keys.setdefault((n[0], comp_of[g.pt[k]]), []).append(k)
    uv = np.zeros((len(g.nodes), 2))
    col = np.zeros((len(g.nodes), 3))
    for (sym, ci), ks in keys.items():
        ks = np.array(ks)
        if g.is_card:
            uv[ks] = _rnd(R[ks, 4:6], 32)
        else:
            u0, v0 = _rnd(R[ks, 4:6].min(0), 32)
            u1, v1 = _rnd(R[ks, 4:6].max(0), 32)
            allk = np.where(comp_of[g.pt] == ci)[0]
            Q = Pn[allk]
            ext = np.ptp(Q, 0)
            ax = np.argsort(-ext)[:2]
            q0, q1 = Q[:, ax].min(0), np.maximum(Q[:, ax].max(0), Q[:, ax].min(0) + 1e-6)
            tt = (Pn[ks][:, ax] - q0) / (q1 - q0)
            uv[ks, 0] = u0 + tt[:, 0] * (u1 - u0)
            uv[ks, 1] = v0 + tt[:, 1] * (v1 - v0)
        col[ks] = _rnd(R[ks, 6:9].mean(0), 32)
    for k, n in enumerate(g.nodes):
        r = g.rows[k]
        if lit_of[n[0]]:
            nb = np.round(nrm[k] * 120).astype(int) % 256
            rgb = tuple(int(x) for x in nb)
        else:
            sh = 0.55 + 0.45 * max(0.0, float(np.dot(nrm[k], LIGHT)))
            rgb = tuple(int(min(255, max(0, x * sh))) for x in col[k])
        a = int(min(255, _rnd(r[9], 85)))
        out[n] = (int(Pn[k, 0]), int(Pn[k, 1]), int(Pn[k, 2]), r[3],
                  int(np.clip(round(uv[k, 0]), -32768, 32767)), int(np.clip(round(uv[k, 1]), -32768, 32767)),
                  rgb[0], rgb[1], rgb[2], a)
    return out


def _score(g, variant):
    """Gate score of one variant (tolerant voxel IoU vs retail, fail flag)."""
    from cleanroom.originality import geometry
    tris = np.array([[g.nidx[x] for x in t] for t in g.tris_n])
    P = np.round(shape_group(g, variant)[g.pt])
    for attempt in range(6):        # the PC is shared: ride out short commit-limit spikes
        try:
            r = geometry.compare((g.retail, tris), (P, tris))
            return r["voxel_iou"], r["fail"]
        except MemoryError:
            import gc
            import time
            gc.collect()
            time.sleep(10)
    raise MemoryError("geometry.compare kept failing for " + g.name)


def choose_variants(groups, log=None):
    """Variant per group. Retail is only a veto here: designs come from our styles; the
    first variant (in a fixed order) whose every LOD of the machine stays under PASS_IOU
    is used; failing that the best for the key, then per-LOD fallback."""
    keys = {}
    for g in groups:
        if not g.is_card:
            keys.setdefault((g.kind, g.style_key), []).append(g)
    choice = {}
    for k, gs in keys.items():
        cache = {}

        def sc(g, v):
            if (g.name, v) not in cache:
                cache[(g.name, v)] = _score(g, v)
            return cache[(g.name, v)]
        best, best_w = 0, 9
        # diagonal spindles only for tiny pieces and objects, never for a whole machine
        nv = N_VARIANTS if (k[0] == "object" or max(g.npt for g in gs) <= 12) else 3 * N_BASE
        for v in range(nv):
            w = 0
            for g in gs:
                iou, fail = sc(g, v)
                w = max(w, iou + (1 if fail else 0))
                if w > best_w:
                    break
            if w < best_w:
                best, best_w = v, w
            if best_w <= PASS_IOU:
                break
        for g in gs:
            choice[g.name] = best
            iou, fail = sc(g, best)
            if fail or iou > 0.38:
                vs = sorted(range(nv), key=lambda v: sc(g, v)[0] + sc(g, v)[1])
                choice[g.name] = vs[0]
        if log:
            log(k, best, round(best_w, 3))
    return choice


def build(dirty_tree, select=True):
    topo = Topo(dirty_tree)
    lit_of = {s: _is_lit(rows) for s, rows in topo.arrays.items()}
    groups = [Group(topo, name) for name in topo.group_order]
    choice = choose_variants(groups) if select else {}
    new = {}
    for g in groups:
        new.update(rows_for_group(g, lit_of, choice.get(g.name, 0)))
    return topo, groups, new, choice


# ------------------------------------------------------------------ C text
def fmt_row(indent, r):
    return "%s{{{ %5d, %5d, %5d }, %d, { %5d, %5d }, { %3d, %3d, %3d, %3d }}}," % ((indent,) + tuple(r))


def _rewrite_file(path, new):
    text = open(path, encoding="utf8", newline="").read()
    out, pos, n_arr, n_vtx = [], 0, 0, 0
    for m in ARRAY_RE.finditer(text):
        sym = m.group(1)
        lines = m.group(2).split("\n")
        k = 0
        for li, line in enumerate(lines):
            v = VTX_LINE.match(line.rstrip("\r"))
            if v:
                key = (sym, k)
                if key in new:
                    cr = "\r" if line.endswith("\r") else ""
                    lines[li] = fmt_row(v.group(1), new[key]) + cr
                    n_vtx += 1
                k += 1
        out.append(text[pos:m.start(2)])
        out.append("\n".join(lines))
        pos = m.end(2)
        n_arr += 1
    out.append(text[pos:])
    open(path, "w", encoding="utf8", newline="").write("".join(out))
    return n_arr, n_vtx


def write(dirty_tree, clean_tree):
    topo, groups, new, choice = build(dirty_tree)
    summ = {"arrays": 0, "vertices": 0, "groups": len(groups),
            "card_groups": sum(g.is_card for g in groups), "meshes": sum(not g.is_card for g in groups)}
    for f in FILES:
        a, v = _rewrite_file(os.path.join(clean_tree, ASSET_DIR, f, f + ".c"), new)
        summ["arrays"] += a
        summ["vertices"] += v
    summ["spindle_groups"] = sorted(n for n, v in choice.items() if v >= 3 * N_BASE)
    summ["variant_histogram"] = {str(v): list(choice.values()).count(v) for v in sorted(set(choice.values()))}
    summ["untouched_vertices"] = sum(len(r) for r in topo.arrays.values()) - summ["vertices"]
    return summ


# ------------------------------------------------------------------ check / gate
def _strip_vtx(path):
    return [l for l in open(path, encoding="utf8").read().splitlines() if not VTX_LINE.match(l)]


def check(dirty_tree, clean_tree):
    res = {"files": {}, "ok": True}
    td, tc = Topo(dirty_tree), Topo(clean_tree)
    card_arrays = set()
    for name in td.group_order:
        g = Group(td, name)
        if g.is_card:
            card_arrays |= {n[0] for n in g.nodes}
    for f in FILES:
        pd = os.path.join(dirty_tree, ASSET_DIR, f, f + ".c")
        pc = os.path.join(clean_tree, ASSET_DIR, f, f + ".c")
        ad, ac = read_arrays(pd), read_arrays(pc)
        same_names = list(ad) == list(ac)
        same_counts = all(len(ad[s]) == len(ac.get(s, [])) for s in ad)
        same_flags = all(a[3] == b[3] for s in ad for a, b in zip(ad[s], ac.get(s, [])))
        same_text = _strip_vtx(pd) == _strip_vtx(pc)
        unchanged = [s for s in ad if ad[s] == ac.get(s) and s not in card_arrays]
        moved = [s for s in ad if s not in card_arrays and
                 np.mean([a[:3] != b[:3] for a, b in zip(ad[s], ac[s])]) < 0.5]
        ok = same_names and same_counts and same_flags and same_text and not unchanged
        res["files"][f] = {"arrays": len(ad), "names": same_names, "counts": same_counts, "flags": same_flags,
                           "dl_text_equal": same_text, "unchanged": unchanged[:5], "cards": len(card_arrays & set(ad)),
                           "mostly_same_pos": len(moved), "ok": ok}
        res["ok"] &= ok
    res["tris_equal"] = td.group_tris == tc.group_tris
    res["ok"] &= res["tris_equal"]
    return res


def export_gate(dirty_tree, clean_tree, gate_dir):
    td, tc = Topo(dirty_tree), Topo(clean_tree)
    n = 0
    for name in td.group_order:
        g = Group(td, name)
        if g.is_card:
            continue
        key = name.split(":", 1)[1]
        for side, topo in (("retail_geo", td), ("clean_geo", tc)):
            verts = [list(topo.arrays[s][i][:3]) for s, i in g.nodes]
            tris = [[g.nidx[x] for x in t] for t in g.tris_n]
            d = os.path.join(gate_dir, side, g.file)
            os.makedirs(d, exist_ok=True)
            json.dump({"verts": verts, "tris": tris}, open(os.path.join(d, key + ".json"), "w"))
        n += 1
    return n


def main(argv):
    if argv[0] == "write":
        print(json.dumps(write(argv[1], argv[2]), indent=1))
    elif argv[0] == "check":
        r = check(argv[1], argv[2])
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    elif argv[0] == "gate":
        print("exported", export_gate(argv[1], argv[2], argv[3]), "meshes")


if __name__ == "__main__":
    main(sys.argv[1:])


