"""CLEAN ROOM: new course layouts for all 26 course slots (24 Grand Prix courses, the battle loop, the ending
parade loop), generated from our own design table, never from retail control points.

Kept from retail (spec/courses.json, structural only): per slot the venue (which scenery/texture set the code
loads), skybox, music slot, creator id and flag. Everything that shapes a course is ours: the loop
(random-harmonic or figure-eight plan, stretched/rotated, resampled into our own control-point count), the
height profile, track widths, segment shapes (road styles, walls, borderless, tunnels, pipes, cylinders,
half-pipes, air gaps), banking and every feature lane (pits, dash plates, dirt, ice, jump plates, mines,
gates, buildings, signs).

Validity is checked here with the game's own spline (games/fzerox/spline.py: tensions inside [0, 2]) and
the join-length rule of Course_SegmentJoinsInit (a non-joinable shape next to a joinable one needs
length > join length + 100), plus our own safety margins: turn radius, spacing, no track crossing another
within reach unless it is a bridge (height gap), bounds of the course editor (|x|,|z| <= 15000,
-250 <= y <= 5000). Raceability is then proven in the emulator (tools/race_check.py).

    python -m games.fzerox.trackgen preview <out.png>   # top views of all 26 layouts (dev)
    python -m games.fzerox.trackgen summary
"""
import json
import math
import os
import sys

import numpy as np

from . import spline
from .courses import FLAG_8000000, FLAG_INSIDE, FLAG_JOINABLE, LANES, checksum

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec", "courses.json")

ROAD, WALLED, PIPE, CYL, HALF, TUNNEL, AIR, BORDERLESS = 0x0, 0x40, 0x80, 0xC0, 0x100, 0x140, 0x180, 0x1C0
JOINABLE = {ROAD, AIR}
JOIN_LEN = {ROAD: 0, WALLED: 50, PIPE: 1000, CYL: 1000, HALF: 1000, TUNNEL: 250, AIR: 0, BORDERLESS: 50}
FLAGS = {ROAD: FLAG_JOINABLE | FLAG_8000000, WALLED: FLAG_8000000, PIPE: FLAG_INSIDE, CYL: 0, HALF: 0,
         TUNNEL: FLAG_INSIDE | FLAG_8000000, AIR: FLAG_JOINABLE, BORDERLESS: FLAG_8000000}
TYPES = {ROAD: [2, 3, 4, 5, 6], WALLED: [0, 1, 2], PIPE: [0, 1, 2, 3], CYL: [0, 1, 2, 3], HALF: [0, 1, 2, 3],
         TUNNEL: [0, 1, 2, 3], BORDERLESS: [0, 1], AIR: [0x3F]}
NAMES = {ROAD: "road", WALLED: "walled", PIPE: "pipe", CYL: "cylinder", HALF: "halfpipe", TUNNEL: "tunnel",
         AIR: "air", BORDERLESS: "borderless"}

# Our design table: one brief per slot. plan = loop kind; lap = target lap length; hills = height amplitude;
# specials = shaped runs to place (kind, count); feat = feature flavour.
DESIGN = [
    dict(plan="turtle", lap=60000, hills=300, specials=[(WALLED, 1)], feat="dash"),
    dict(plan="loop", lap=72000, hills=150, specials=[(TUNNEL, 2)], feat="dash", stretch=1.8),
    dict(plan="turtle", lap=66000, hills=500, specials=[(PIPE, 1)], feat="dash"),
    dict(plan="turtle", lap=62000, hills=700, specials=[(WALLED, 1)], feat="dirt", wiggle=1.3),
    dict(plan="turtle", lap=76000, hills=1200, specials=[(CYL, 1)], feat="dash"),
    dict(plan="turtle", lap=70000, hills=900, specials=[(AIR, 1)], feat="ice"),
    dict(plan="turtle", lap=68000, hills=1500, specials=[(TUNNEL, 1), (BORDERLESS, 1)], feat="mixed"),
    dict(plan="turtle", lap=64000, hills=600, specials=[(HALF, 1)], feat="dash"),
    dict(plan="turtle", lap=70000, hills=1100, specials=[(BORDERLESS, 2)], feat="dirt", wiggle=1.4),
    dict(plan="eight", lap=74000, hills=700, specials=[(WALLED, 1)], feat="dash"),
    dict(plan="turtle", lap=78000, hills=2000, specials=[(AIR, 2)], feat="jump"),
    dict(plan="turtle", lap=72000, hills=800, specials=[(PIPE, 1), (TUNNEL, 1)], feat="mixed"),
    dict(plan="turtle", lap=66000, hills=500, specials=[(WALLED, 2)], feat="dirt", wiggle=1.5),
    dict(plan="turtle", lap=80000, hills=1600, specials=[(CYL, 1), (BORDERLESS, 1)], feat="dash"),
    dict(plan="turtle", lap=74000, hills=900, specials=[(HALF, 1), (AIR, 1)], feat="jump"),
    dict(plan="turtle", lap=70000, hills=700, specials=[(WALLED, 2)], feat="dirt"),
    dict(plan="turtle", lap=76000, hills=1300, specials=[(BORDERLESS, 2)], feat="ice"),
    dict(plan="loop", lap=86000, hills=400, specials=[(TUNNEL, 1)], feat="dash", stretch=2.0),
    dict(plan="eight", lap=82000, hills=900, specials=[(PIPE, 1)], feat="mixed"),
    dict(plan="turtle", lap=78000, hills=2400, specials=[(BORDERLESS, 2)], feat="dash", wiggle=1.3),
    dict(plan="turtle", lap=84000, hills=1000, specials=[(CYL, 2)], feat="dash"),
    dict(plan="turtle", lap=80000, hills=1800, specials=[(AIR, 2), (WALLED, 1)], feat="jump"),
    dict(plan="eight", lap=82000, hills=1200, specials=[(TUNNEL, 1), (HALF, 1)], feat="mixed"),
    dict(plan="turtle", lap=88000, hills=2000, specials=[(BORDERLESS, 2), (PIPE, 1)], feat="mixed", wiggle=1.3),
    dict(plan="loop", lap=30000, hills=300, specials=[], feat="battle"),
    dict(plan="loop", lap=58000, hills=0, specials=[], feat="parade", stretch=2.2),
]
VENUE_WINTER, VENUE_DESERT, VENUE_CANYON, VENUE_FOREST, VENUE_FIRE = 5, 3, 7, 4, 8


def load_spec():
    return json.load(open(SPEC, encoding="utf8"))


# ------------------------------------------------------------------ plan

def _plan_turtle(d, rng):
    """Straights and arcs (turtle walk) that turn a full circle, closed by spreading the end gap over the walk."""
    nt = int(rng.integers(5, 10))
    turn = rng.uniform(0.4, 1.6, nt)
    sign = np.where(rng.uniform(size=nt) < d.get("reverse", 0.22), -0.6, 1.0)
    turn *= sign
    if rng.uniform() < d.get("hairpin", 0.35):
        turn[int(rng.integers(0, nt))] = rng.uniform(2.6, 3.1)
    if turn.sum() < 2.5:
        turn = np.abs(turn)
    turn *= 2 * np.pi / turn.sum()
    radius = rng.uniform(0.6, 2.4, nt)
    straight = rng.uniform(0.3, 2.5, nt) * d.get("straights", 1.0)
    pts, h, p = [], 0.0, np.zeros(2)
    for k in range(nt):
        for _ in range(max(2, int(straight[k] * 20))):
            p = p + 0.05 * np.array([math.cos(h), math.sin(h)])
            pts.append(p)
        steps = max(4, int(abs(turn[k]) * radius[k] * 20))
        for _ in range(steps):
            h += turn[k] / steps
            p = p + (abs(turn[k]) * radius[k] / steps) * np.array([math.cos(h), math.sin(h)])
            pts.append(p)
    pts = np.array(pts)
    gap = pts[-1] - pts[0] + (pts[1] - pts[0])
    seg = np.linalg.norm(np.diff(pts, axis=0, prepend=pts[:1]), axis=1)
    frac = np.cumsum(seg) / seg.sum()
    pts = pts - frac[:, None] * gap
    # smooth the joins a little (light moving average, closed)
    for _ in range(3):
        pts = (np.roll(pts, 1, 0) + 2 * pts + np.roll(pts, -1, 0)) / 4
    return np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]], 1)


def _plan(d, rng):
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    if d["plan"] == "turtle":
        P = _plan_turtle(d, rng)
        s = d.get("stretch", 1.0) * rng.uniform(0.9, 1.15)
        P[:, 0] *= s
        ang = rng.uniform(0, 2 * np.pi)
        x, z = P[:, 0].copy(), P[:, 2].copy()
        P[:, 0], P[:, 2] = x * math.cos(ang) - z * math.sin(ang), x * math.sin(ang) + z * math.cos(ang)
        seglen = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        return P * (d["lap"] / seglen.sum()), None
    if d["plan"] == "eight":
        a = 1.0 + 0.15 * rng.uniform(-1, 1)
        x = np.cos(th) * a
        z = np.sin(th) * np.cos(th) * (1.2 + 0.3 * rng.uniform(-1, 1))
        for k in (3, 5):
            x += 0.05 * rng.uniform(-1, 1) * np.cos(k * th + rng.uniform(0, 6.3))
            z += 0.05 * rng.uniform(-1, 1) * np.sin(k * th + rng.uniform(0, 6.3))
        bridge = np.sin(th)            # +1 / -1 at the two passes through the crossing
    else:
        r = np.ones_like(th)
        w = d.get("wiggle", 1.0)
        for k in range(2, 7):
            amp = rng.uniform(0.0, 0.22 * w) / (k ** 0.8)
            r += amp * np.cos(k * th + rng.uniform(0, 2 * np.pi))
        r = np.maximum(r, 0.35)
        x, z = r * np.cos(th), r * np.sin(th)
        bridge = None
    s = d.get("stretch", 1.0) * rng.uniform(0.85, 1.25)
    x = x * s
    ang = rng.uniform(0, 2 * np.pi)
    x, z = x * math.cos(ang) - z * math.sin(ang), x * math.sin(ang) + z * math.cos(ang)
    P = np.stack([x, np.zeros_like(x), z], 1)
    seglen = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    P *= d["lap"] / seglen.sum()
    return P, bridge


def _heights(d, rng, n, bridge):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = np.zeros(n)
    for k in range(1, 4):
        y += rng.uniform(0.2, 1.0) / k * np.sin(k * s + rng.uniform(0, 2 * np.pi))
    if np.ptp(y) > 0:
        y = (y - y.min()) / np.ptp(y) * d["hills"]
    if bridge is not None:
        y = y * 0.5 + (bridge + 1.0) * 550.0     # 1100 apart where the figure eight crosses itself
    return y + 40.0


def _resample(P, n, rng):
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    u = np.arange(n) + rng.uniform(-0.18, 0.18, n)
    u[0] = 0
    targets = np.sort(u) / n * L
    out = []
    for t in targets:
        i = np.searchsorted(cum, t, side="right") - 1
        f = (t - cum[i]) / max(seg[i % len(seg)], 1e-9)
        out.append(P[i % len(P)] * (1 - f) + P[(i + 1) % len(P)] * f)
    return np.array(out), targets / L


# ------------------------------------------------------------------ checks

def _turn_radius(pts):
    d1 = (np.roll(pts, -1, 0) - np.roll(pts, 1, 0)) * 0.5
    d2 = np.roll(pts, -1, 0) - 2 * pts + np.roll(pts, 1, 0)
    k = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    return 1.0 / np.maximum(k, 1e-9), d1, d2


def _crossings_ok(pts, width, clear=900.0):
    n = len(pts)
    step = 4
    q = pts[::step]
    w = width[::step]
    cum = np.arange(len(q)) * step
    for i in range(len(q)):
        dh = np.hypot(q[:, 0] - q[i, 0], q[:, 2] - q[i, 2])
        dv = np.abs(q[:, 1] - q[i, 1])
        sep = np.minimum(np.abs(cum - cum[i]), n - np.abs(cum - cum[i]))
        bad = (sep > 150) & (dh < w + w[i] + 500) & (dv < clear)
        if bad.any():
            return False
    return True


# ------------------------------------------------------------------ build one course

def build(slot, seed, spec_slot):
    d = DESIGN[slot]
    rng = np.random.default_rng([0x6C1D, slot, seed])
    plan, bridge = _plan(d, rng)
    n = int(spec_slot["count"])          # the slot's control-point count (structure: buffers sized by it)
    P, frac = _resample(plan, n, rng)
    if bridge is not None:
        idx = (frac * len(bridge)).astype(int) % len(bridge)
        y = 40.0 + (bridge[idx] + 1.0) * 550.0 + _heights(dict(d, hills=d["hills"] * 0.5), rng, n, None)
    else:
        y = _heights(d, rng, n, None)
    P[:, 1] = y
    # centre on the origin, inside the editor bounds
    P[:, [0, 2]] -= (P[:, [0, 2]].max(0) + P[:, [0, 2]].min(0)) / 2
    if np.abs(P[:, [0, 2]]).max() > 14500 or P[:, 1].max() > 4900:
        return None, "bounds"
    a1, a2, ok = spline.tensions(P)
    if not ok.all():
        return None, "tension"
    pts, seg, T, _ = spline.curve(P, 20)
    rad, d1, d2 = _turn_radius(pts)
    if rad.min() < 480:
        return None, f"turn {rad.min():.0f}"
    L = spline.lengths(P)
    if L.min() < 800:
        return None, "short segment"
    # widths: our own band per course, narrower in tight turns
    base = rng.uniform(190, 300)
    rmin_seg = np.array([rad[seg == i].min() for i in range(n)])
    width = np.clip(base * np.clip(rmin_seg / 1500, 0.75, 1.25), 150, 380)
    # shapes
    shape = np.full(n, ROAD)
    hold = {0, 1, n - 1}               # start straight stays plain road

    def free(i0, k):
        return all(((i0 + j) % n) not in hold and shape[(i0 + j) % n] == ROAD for j in range(-1, k + 1))

    for kind, count in d["specials"]:
        for _ in range(count):
            for _try in range(200):
                if kind == AIR:
                    k = 1
                elif kind in (PIPE, CYL, HALF):
                    k = int(rng.integers(3, 6))
                else:
                    k = int(rng.integers(2, 5))
                i0 = int(rng.integers(3, n - k - 2))
                if not free(i0, k):
                    continue
                run = [(i0 + j) % n for j in range(k)]
                need = 2 * JOIN_LEN[kind] + 400 if k == 1 else JOIN_LEN[kind] + 400
                if L[run[0]] < need or L[run[-1]] < need:
                    continue
                if kind == AIR:
                    # gaps only where the course drops: machines land downhill
                    if P[(i0 + 1) % n, 1] > P[i0, 1] - 150 or rmin_seg[i0] < 2500:
                        continue
                if kind in (PIPE, CYL, HALF) and min(rmin_seg[r] for r in run) < 1100:
                    continue
                for r in run:
                    shape[r] = kind
                    hold.add(r)
                break
    widths_l, widths_r = width.copy(), width.copy()
    for i in range(n):
        if shape[i] in (PIPE, CYL, HALF):
            r = {PIPE: rng.uniform(230, 300), CYL: rng.uniform(260, 330), HALF: rng.uniform(260, 320)}[shape[i]]
            widths_l[i] = widths_r[i] = r
    # keep special-shape runs at one radius
    for i in range(n):
        if shape[i] in (PIPE, CYL, HALF) and shape[i - 1] == shape[i]:
            widths_l[i] = widths_r[i] = widths_l[i - 1]
    if not _crossings_ok(pts, np.repeat(np.maximum(widths_l, widths_r), 20)):
        return None, "crossing"
    # segment types: our own alternation per course
    road_pair = rng.choice([2, 3, 4, 5, 6], 2, replace=False)
    info = np.zeros(n, np.int64)
    for i in range(n):
        s = shape[i]
        if s == ROAD:
            t = 0 if i == 0 else int(road_pair[i % 2])
        else:
            opts = TYPES[s]
            t = int(opts[(slot + i) % len(opts)])
        info[i] = FLAGS[s] | s | t
    # banking from the horizontal turn (sign as the game expects: inside of the turn lower)
    bank = np.zeros(n, int)
    bmax = rng.uniform(18, 34)
    for i in range(n):
        if shape[i] not in (ROAD, WALLED, BORDERLESS):
            continue
        j = i * 20
        k = (d1[j, 0] * d2[j, 2] - d1[j, 2] * d2[j, 0]) / max(np.hypot(d1[j, 0], d1[j, 2]) ** 3, 1e-9)
        b = float(np.clip(k * 9000.0, -1, 1)) * bmax * (1.6 if shape[i] == BORDERLESS else 1.0)
        bank[i] = int(round(b)) % 360
    bank[0] = bank[1] = bank[n - 1] = 0
    lanes = _features(slot, d, rng, n, shape, L, rmin_seg, spec_slot["venue"])
    cps = [[float(round(P[i, 0])), float(round(P[i, 1])), float(round(P[i, 2])), int(widths_l[i]), int(widths_r[i]),
            int(info[i])] for i in range(n)]
    c = dict(name=spec_slot["name"], creator=spec_slot["creator"], count=n, venue=spec_slot["venue"],
             skybox=spec_slot["skybox"], flag=spec_slot["flag"], bgm=spec_slot["bgm"],
             filename=list(b"GW%02d" % slot) + [0] * 18,
             cps=cps + [[0.0, 0.0, 0.0, 0, 0, 0]] * (64 - n), bank=list(bank) + [0] * (64 - n))
    for k in LANES:
        c[k] = list(lanes[k]) + [0] * (64 - n)
    plain = os.environ.get("GW_PLAIN", "")          # DEV bisect: "shapes", "lanes", "bank" stripped
    if "shapes" in plain:
        for i in range(n):
            t = 0 if i == 0 else 2 + i % 2
            c["cps"][i][5] = FLAGS[ROAD] | ROAD | t
            c["cps"][i][3] = c["cps"][i][4] = 260
    if "lanes" in plain:
        for k in LANES:
            c[k] = [-1] * n + [0] * (64 - n)
        c["gate"][1] = 1
    if "bank" in plain:
        c["bank"] = [0] * 64
    c["checksum"] = checksum(c)
    c["_cpu_len"] = spec_slot.get("cpu_len")
    c["_shape"] = [NAMES[s] for s in shape]
    c["_lap"] = float(L.sum())
    return c, "ok"


def _features(slot, d, rng, n, shape, L, rmin, venue):
    f = {k: np.full(n, -1, int) for k in LANES}
    flat = [i for i in range(2, n - 1) if shape[i] in (ROAD, WALLED, BORDERLESS) and shape[i - 1] == shape[i]
            and shape[(i + 1) % n] == shape[i]]
    straight = [i for i in flat if rmin[i] > 2500]
    pick = lambda pool, k: list(rng.choice(pool, min(k, len(pool)), replace=False)) if pool else []
    f["gate"][1] = 1                                    # start gate (the code expects it at control point 1)
    if d["feat"] == "parade":
        for i in pick(flat, 8):
            f["building"][i] = int(rng.choice([1, 2, 4, 5, 7, 8]))
        for i in pick(flat, 6):
            f["sign"][i] = int(rng.choice([0, 1, 2]))
        return f
    pits = pick(straight or flat, int(rng.integers(1, 3)))
    for i in pits:
        f["pit"][i] = int(rng.choice([0, 0, 1, 2]))
    rest = [i for i in flat if i not in pits]
    n_dash = {"dash": int(rng.integers(4, 9)), "battle": 2}.get(d["feat"], int(rng.integers(2, 5)))
    for i in pick(rest, n_dash):
        f["dash"][i] = int(rng.choice([0, 1, 2]))
    for i in range(n):
        if shape[i] in (CYL,) and rng.uniform() < 0.3:
            f["dash"][i] = 0
    free = [i for i in rest if f["dash"][i] == -1]
    if d["feat"] in ("dirt", "mixed") or venue in (VENUE_DESERT, VENUE_CANYON, VENUE_FOREST):
        for i in pick(free, int(rng.integers(1, 4))):
            f["dirt"][i] = int(rng.choice([0, 1, 2, 3]))
    free = [i for i in free if f["dirt"][i] == -1]
    if d["feat"] in ("ice",) or venue == VENUE_WINTER:
        for i in pick(free, int(rng.integers(1, 4))):
            f["ice"][i] = int(rng.choice([0, 1, 2, 3]))
    free = [i for i in free if f["ice"][i] == -1]
    if d["feat"] in ("jump", "mixed"):
        before_air = [i for i in range(2, n) if shape[(i + 1) % n] == AIR and shape[i] == ROAD]
        for i in before_air + pick([i for i in free if rmin[i] > 3000], 1 if d["feat"] == "jump" else 0):
            f["jump"][i] = 0
    if d["feat"] == "mines":
        for i in pick([i for i in free if f["jump"][i] == -1], int(rng.integers(2, 4))):
            f["landmine"][i] = int(rng.choice([0, 1, 2]))
    gates = [i for i in flat if i > 3 and f["pit"][i] == -1]
    for i in pick(gates, int(rng.integers(1, 4))):
        f["gate"][i] = int(rng.choice([0, 2]))
    side = [i for i in range(2, n) if shape[i] in (ROAD, WALLED, BORDERLESS)]
    for i in pick(side, int(rng.integers(2, 7))):
        f["building"][i] = int(rng.integers(0, 15))
    for i in pick(side, int(rng.integers(2, 8))):
        if f["building"][i] == -1:
            f["sign"][i] = int(rng.choice([0, 1, 2, 3, 4]))
    return f


def seeds():
    p = os.path.join(HERE, "spec", "course_seeds.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def generate_all(verbose=False, start_seeds=None):
    spec = load_spec()
    sd = dict(seeds())
    sd.update(start_seeds or {})
    out = []
    for slot in range(len(spec["slots"])):
        seed = int(sd.get(str(slot), 0))
        reasons = {}
        for tries in range(4000):
            c, why = build(slot, seed, spec["slots"][slot])
            if c is not None:
                break
            reasons[why.split()[0]] = reasons.get(why.split()[0], 0) + 1
            seed += 1
        else:
            raise SystemExit(f"slot {slot}: no valid layout ({reasons})")
        c["_seed"] = seed
        out.append(c)
        if verbose:
            print(f"slot {slot:2d} seed {seed:4d} n={c['count']:2d} lap={c['_lap']:6.0f} "
                  f"{ {s: c['_shape'].count(s) for s in set(c['_shape'])} } rejected {reasons}")
    return out


def cpu_table(c, slot):
    """CPU guidance for one course (cpu.c loads 0x200 s16 per Grand Prix course): per half segment a speed cap
    and a lateral offset. Ours: cap 70 on open road, lower in our tight turns; a gentle weave of the line
    (|offset| <= 60) on roads, centred in tubes, drums and bowls. Size = 4 * points + 8 (the slot's layout)."""
    n = c["count"]
    length = c.get("_cpu_len") or 4 * n + 8
    if CPU_MODE == "flat":
        # the game's own guidance for courses without data (cpu.c fills 69 / 0 for edit courses)
        return [69, 0] * (length // 2)
    P = np.array([p[:3] for p in c["cps"][:n]])
    pts, seg, T, _ = spline.curve(P, 20)
    rad, _, _ = _turn_radius(pts)
    rng = np.random.default_rng([0xC0, slot])
    ph, m = rng.uniform(0, 2 * np.pi), int(rng.integers(2, 5))
    vals = []
    for h in range(2 * n):
        i = h // 2
        r = rad[h * 10:(h + 1) * 10].min()
        cap = int(np.clip(52 + 18 * (r - 500) / 1500, 52, 70))
        shape = c["cps"][i][5] & 0x1C0
        off = 0 if shape in (PIPE, CYL, HALF, AIR) else int(round(60 * math.sin(ph + 2 * np.pi * m * h / (2 * n))))
        vals += [cap, off]
    vals += [69, 0] * ((length - len(vals)) // 2)
    return vals[:length]


CPU_MODE = os.environ.get("GW_CPU_MODE", "flat")


def write_cpu_tables(out, cs, ver="us/rev0"):
    cpu = open(os.path.join(out, "src", "overlays", "ovl_i3", "cpu.c"), encoding="utf8").read()
    blk = cpu[cpu.index("D_i3_8013DB80[] = {"):]
    names = [x for x in __import__("re").findall(r"D_F2[0-9A-F]+", blk[:blk.index("};")])]
    p = os.path.join(out, "src", "assets", ver, "common_assets_compressed", "common_assets_compressed.c")
    s = open(p, encoding="utf8").read()
    import re
    for slot, nm in enumerate(names[:24]):
        vals = cpu_table(cs[slot], slot)
        body = "\n" + "\n".join("    " + ", ".join(str(v) for v in vals[i:i + 10]) + ","
                                for i in range(0, len(vals), 10)) + "\n"
        s, k = re.subn(r"(s16 " + nm + r"\[\] = \{)[^}]*(\};)", lambda m: m.group(1) + body + m.group(2), s)
        assert k == 1, nm
    open(p, "w", encoding="utf8", newline="\n").write(s)
    print(f"cpu guidance tables: {len(names[:24])} regenerated")


def write(out, ver="us/rev0"):
    """Clean tree: our course_data.c (all 26 slots) + the gate meshes of our layouts."""
    from .courses import emit, ribbon, save_mesh
    cs = generate_all()
    p = os.path.join(out, "src", "assets", ver, "course_data", "course_data.c")
    with open(p, "w", newline="\n") as f:
        f.write(emit([{k: v for k, v in c.items() if not k.startswith("_")} for c in cs]))
    for c in cs:
        v, t = ribbon(c)
        save_mesh(os.path.join(out, "gate", "courses", c["name"] + ".json"), v, t)
    json.dump([{"slot": i, "seed": c["_seed"], "count": c["count"], "lap": round(c["_lap"]), "shapes": c["_shape"]}
               for i, c in enumerate(cs)], open(os.path.join(out, "gate", "courses.json"), "w"), indent=0)
    print(f"courses: {len(cs)} new layouts written (seeds {[c['_seed'] for c in cs]})")
    write_cpu_tables(out, cs, ver)
    return cs


def main(argv):
    if argv[1] == "summary":
        generate_all(verbose=True)
    elif argv[1] == "preview":
        from PIL import Image, ImageDraw
        cs = generate_all()
        W = 260
        im = Image.new("RGB", (W * 7, W * 4), (20, 20, 24))
        dr = ImageDraw.Draw(im)
        col = {"road": (220, 220, 220), "walled": (120, 200, 255), "pipe": (255, 120, 60), "cylinder": (255, 200, 40),
               "halfpipe": (120, 255, 120), "tunnel": (160, 80, 220), "air": (255, 40, 40), "borderless": (90, 140, 90)}
        for k, c in enumerate(cs):
            n = c["count"]
            P = np.array([p[:3] for p in c["cps"][:n]])
            pts, seg, T, _ = spline.curve(P, 10)
            ox, oy = (k % 7) * W + W // 2, (k // 7) * W + W // 2
            sc = (W - 30) / 30000
            for j in range(len(pts)):
                a, b = pts[j], pts[(j + 1) % len(pts)]
                dr.line([(ox + a[0] * sc, oy + a[2] * sc), (ox + b[0] * sc, oy + b[2] * sc)],
                        fill=col[c["_shape"][seg[j]]], width=3)
            dr.text((ox - W // 2 + 4, oy - W // 2 + 4), f"{k} n{n} {c['_lap'] / 1000:.0f}k", fill=(255, 255, 0))
        im.save(argv[2])
        print("preview ->", argv[2])


if __name__ == "__main__":
    main(sys.argv)
