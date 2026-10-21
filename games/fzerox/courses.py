"""Course data (CourseData, include/fzx_course.h): parse / emit the Torch C, checksum, summaries.

    python -m games.fzerox.courses summary <course_data.c> <include dir>     # one line per course
    python -m games.fzerox.courses export <course_data.c> <include dir> <out dir>   # DIRTY side: gate meshes

A course is 64 control-point slots (position, left/right radius, segment info = shape/type/flags), a bank
angle per point and nine per-point feature lanes (pit, dash, dirt, ice, jump, landmine, gate, building,
sign). The game builds the track mesh at load time from these points, so a course is fully described by
this record.
"""
import json
import os
import re
import sys

import numpy as np

LANES = ["pit", "dash", "dirt", "ice", "jump", "landmine", "gate", "building", "sign"]
SHAPES = {0x0: "road", 0x40: "walled", 0x80: "pipe", 0xC0: "cylinder", 0x100: "halfpipe", 0x140: "tunnel",
          0x180: "air", 0x1C0: "borderless"}
SHAPE_CODE = {v: k for k, v in SHAPES.items()}
FLAG_8000000, FLAG_JOINABLE, FLAG_INSIDE = 0x8000000, 0x10000000, 0x20000000


def header_values(include_dir):
    """Identifier -> int from fzx_course.h (#defines and enums)."""
    txt = open(os.path.join(include_dir, "fzx_course.h"), encoding="utf8").read()
    vals = {}
    for m in re.finditer(r"#define\s+(\w+)\s+(0x[0-9A-Fa-f]+|-?\d+)\b", txt):
        vals[m.group(1)] = int(m.group(2), 0)
    for body in re.findall(r"typedef enum \w*\s*\{(.*?)\}", txt, re.S):
        body = re.sub(r"#\w+[^\n]*", "", body)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        n = 0
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                k, v = [s.strip() for s in item.split("=")]
                n = int(v, 0)
            else:
                k = item
            vals[k] = n
            n += 1
    return vals


def _eval(expr, vals):
    tot = 0
    for t in expr.split("|"):
        t = t.strip()
        tot |= vals[t] if t in vals else int(t, 0)
    return tot


def parse(path, include_dir):
    vals = header_values(include_dir)
    txt = open(path, encoding="utf8").read()
    out = []
    for m in re.finditer(r"CourseData (\w+) = \{(.*?)\n\};", txt, re.S):
        name, body = m.group(1), m.group(2)
        body = re.sub(r"/\*.*?\*/", "", body)
        head = body.split("{", 1)[0]
        hv = [h.strip() for h in head.split(",") if h.strip()]
        rest = body[len(head):]
        fn = re.match(r"\{([^}]*)\}", rest)
        fname = [int(x.strip().strip("'").encode()[0]) if x.strip().startswith("'") else int(x, 0)
                 for x in fn.group(1).split(",")]
        rest = rest[fn.end():]
        bgm_m = re.match(r"\s*,\s*([^,]+),", rest)
        bgm = _eval(bgm_m.group(1), vals)
        rest = rest[bgm_m.end():]
        # control points block
        cp_start = rest.index("{")
        depth, i = 0, cp_start
        while True:
            if rest[i] == "{":
                depth += 1
            elif rest[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        cpblk = rest[cp_start + 1:i]
        cps = []
        for cm in re.finditer(r"\{\s*\{\s*\{([^}]*)\}\s*\}\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*([^}]*?)\s*\}|\{\s*0\s*\}",
                              cpblk):
            if cm.group(1) is None:
                cps.append([0.0, 0.0, 0.0, 0, 0, 0])
            else:
                x, y, z = [float(v) for v in cm.group(1).split(",")]
                cps.append([x, y, z, int(cm.group(2)), int(cm.group(3)), _eval(cm.group(4), vals) & 0xFFFFFFFF])
        arrays = re.findall(r"\{([^{}]*)\}", rest[i + 1:])
        arr = [[_eval(v, vals) for v in a.split(",") if v.strip()] for a in arrays]
        c = dict(name=name, creator=_eval(hv[0], vals), count=_eval(hv[1], vals), venue=_eval(hv[2], vals),
                 skybox=_eval(hv[3], vals), checksum=_eval(hv[4], vals) & 0xFFFFFFFF, flag=_eval(hv[5], vals),
                 filename=fname, bgm=bgm, cps=cps, bank=arr[0])
        for k, a in zip(LANES, arr[1:]):
            c[k] = a
        out.append(c)
    return out


# ------------------------------------------------------------------ checksum (Course_CalculateChecksum, US)

def checksum(c):
    f = np.float32
    n = c["count"]
    tot = np.uint32(n)
    with np.errstate(over="ignore"):
        for i in range(n):
            x, y, z, rl, rr, info = c["cps"][i]
            info &= ~(0x600 | 0x38000 | 0x40000000) & 0xFFFFFFFF
            fi = f(i)
            v = (f(x) + (f(1.1) + f(0.7) * fi) * f(y)) + ((f(2.2) + f(1.2) * fi) * f(z) * (f(4.4) + f(0.9) * fi)) \
                + f(rl) + ((f(5.5) + f(0.8) * fi) * f(rr) * f(4.8))
            iv = np.int64(np.trunc(np.float32(v)))
            add = (int(iv) + info * (0xFE - i) + (c["bank"][i] * (0x93DE - i * 2))) & 0xFFFFFFFF
            tot = np.uint32((int(tot) + add) & 0xFFFFFFFF)
        for i in range(n):
            for k, mul in zip(LANES, [0, 0x10, 0x80, 0x100, 0x800, 0x1000, 0x8000, 0x10000, 0x80000]):
                tot = np.uint32((int(tot) + c[k][i] * (i + mul)) & 0xFFFFFFFF)
    return int(tot)


# ------------------------------------------------------------------ emit

def emit(courses):
    lines = ['#include "macros.h"', '#include ASSET_HEADER(course_data.h)', ""]
    for c in courses:
        fn = ", ".join(f"0x{b & 0xFF:02X}" for b in c["filename"])
        lines.append(f"CourseData {c['name']} = {{")
        lines.append(f"    {c['creator']}, {c['count']}, {c['venue']}, {c['skybox']}, 0x{c['checksum']:08X}, {c['flag']},")
        lines.append(f"    {{ {fn} }},")
        lines.append(f"    {c['bgm']},")
        lines.append("    {")
        for p in c["cps"]:
            if not any(p):
                lines.append("        { 0 },")
            else:
                lines.append(f"        {{ {{ {{ {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f} }} }}, {p[3]}, {p[4]}, 0x{p[5] & 0xFFFFFFFF:08X} }},")
        lines.append("    },")
        for k in ["bank"] + LANES:
            lines.append("    { " + ", ".join(str(v) for v in c[k]) + " },")
        lines.append("};")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ geometry for the gate

def centerline(c, samples=12):
    """Closed Catmull-Rom through the used control points -> (N,3) points, (N,) half widths, (N,) bank deg."""
    n = c["count"]
    P = np.array([p[:3] for p in c["cps"][:n]], float)
    R = np.array([(p[3] + p[4]) / 2.0 for p in c["cps"][:n]], float)
    B = np.array([((b + 180) % 360) - 180 for b in c["bank"][:n]], float)
    pts, rad, bank = [], [], []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            rad.append(R[i] * (1 - t) + R[(i + 1) % n] * t)
            bank.append(B[i] * (1 - t) + B[(i + 1) % n] * t)
    return np.array(pts), np.array(rad), np.array(bank)


def ribbon(c, samples=12):
    """Track surface as a triangle ribbon (verts, tris) in world units: what the gate voxelises."""
    pts, rad, bank = centerline(c, samples)
    n = len(pts)
    verts, tris = [], []
    up = np.array([0.0, 1.0, 0.0])
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i - 1]
        d /= np.linalg.norm(d) + 1e-9
        side = np.cross(up, d)
        side /= np.linalg.norm(side) + 1e-9
        a = np.radians(bank[i])
        s2 = side * np.cos(a) + np.cross(d, side) * np.sin(a)
        verts += [list(pts[i] - s2 * rad[i]), list(pts[i] + s2 * rad[i])]
    for i in range(n):
        j = (i + 1) % n
        tris += [[2 * i, 2 * j, 2 * i + 1], [2 * i + 1, 2 * j, 2 * j + 1]]
    return verts, tris


def save_mesh(path, verts, tris):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"verts": [[round(v, 1) for v in p] for p in verts], "tris": tris}, open(path, "w"))


def summary(c):
    n = c["count"]
    P = np.array([p[:3] for p in c["cps"][:n]])
    shapes = [SHAPES[p[5] & 0x1C0] for p in c["cps"][:n]]
    sh = {s: shapes.count(s) for s in sorted(set(shapes))}
    feats = {k: sum(1 for v in c[k][:n] if v != -1) for k in LANES}
    lo, hi = P.min(0), P.max(0)
    radii = [p[3] for p in c["cps"][:n]] + [p[4] for p in c["cps"][:n]]
    return (f"{c['name']:22s} n={n:2d} v={c['venue']:2d} sky={c['skybox']} bgm={c['bgm']:2d} "
            f"x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}] r[{min(radii)},{max(radii)}] "
            f"{sh} {dict((k, v) for k, v in feats.items() if v)}")


def main(argv):
    cmd = argv[1]
    cs = parse(argv[2], argv[3])
    if cmd == "summary":
        for c in cs:
            ok = "ok" if checksum(c) == c["checksum"] else f"cksum {checksum(c):08X}!={c['checksum']:08X}"
            print(summary(c), ok)
    elif cmd == "spec":
        # DIRTY side, once: structural per-slot facts only (symbol, venue, skybox, music slot, creator, flag)
        slots = [dict(name=c["name"], venue=c["venue"], skybox=c["skybox"], bgm=c["bgm"], creator=c["creator"],
                      flag=c["flag"], count=c["count"]) for c in cs]
        if len(argv) > 5:
            # CPU guidance array lengths (cpu.c loads one s16 array per Grand Prix slot; lengths are structure)
            root = argv[5]
            cpu = open(os.path.join(root, "src", "overlays", "ovl_i3", "cpu.c"), encoding="utf8").read()
            blk = cpu[cpu.index("D_i3_8013DB80[] = {"):]
            names = re.findall(r"D_F2[0-9A-F]+", blk[:blk.index("};")])
            ca = open(os.path.join(root, "src", "assets", "us", "rev0", "common_assets_compressed",
                                   "common_assets_compressed.c"), encoding="utf8").read()
            for s, nm in zip(slots, names):
                body = re.search(r"s16 " + nm + r"\[\] = \{([^}]*)\}", ca).group(1)
                s["cpu_array"] = nm
                s["cpu_len"] = len([v for v in body.split(",") if v.strip()])
        json.dump({"note": "Per course slot: which venue/skybox/music the code loads; layouts are ours (trackgen.py)",
                   "slots": slots}, open(argv[4], "w"), indent=1)
        print(f"spec: {len(slots)} slots -> {argv[4]}")
    elif cmd == "export":
        for c in cs:
            v, t = ribbon(c)
            save_mesh(os.path.join(argv[4], c["name"] + ".json"), v, t)
        print(f"exported {len(cs)} course ribbons -> {argv[4]}")


if __name__ == "__main__":
    main(sys.argv)
