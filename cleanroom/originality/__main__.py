"""python -m cleanroom.originality <dirty> <clean> <spec> [--report ORIGINALITY.md] [--manifest F]

One-screen summary; exit status 1 if anything fails (schema: package docstring).
"""
import glob
import json
import os
import sys

import numpy as np

from cleanroom.originality import blockmap, code, geometry, images, music, text, tilemap
from cleanroom.originality.config import merged


def expand(root, pattern):
    """{key: path}; key = path below the pattern's first wildcard."""
    pat = pattern.replace("\\", "/")
    cut = min([pat.find(c) for c in "*?[" if c in pat] or [len(pat)])
    prefix = pat[:cut].rsplit("/", 1)[0] + "/" if "/" in pat[:cut] else ""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        rel = os.path.relpath(p, root).replace("\\", "/")
        out[rel[len(prefix):]] = p
    return out


def load_strings(path):
    if path.endswith(".json"):
        d = json.load(open(path, encoding="utf8"))
        if isinstance(d, dict):
            d = list(d.values())
        return [x["text"] if isinstance(x, dict) else str(x) for x in d]
    return open(path, encoding="utf8", errors="replace").read().splitlines()


def _anchors(entry, base, key):
    if "anchors" not in entry:
        return None
    a = entry["anchors"]
    if isinstance(a, str):
        a = json.load(open(os.path.join(base, a)))
    if isinstance(a, dict):
        a = a.get("by_mesh", {}).get(key, a.get("anchors", []))
    return np.asarray(a, float).reshape(-1, 3)


def _read_img(path):
    """8-bit PNGs via our reader; palette/low-bit-depth (Game Boy, NES) via Pillow as grey."""
    from cleanroom.gfx import png
    try:
        return png.read(path)
    except ValueError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("L"))


class Result:
    def __init__(self):
        self.rows = {}        # kind -> list of (item, score, fail, detail)

    def add(self, kind, item, score, fail, detail):
        self.rows.setdefault(kind, []).append((item, float(score), bool(fail), detail))

    def failing(self):
        return [(k, r) for k, rows in self.rows.items() for r in rows if r[2]]


def run(dirty, clean, man, base, res=None):
    res = res or Result()
    cfg = merged(man.get("thresholds"))

    def pairs(kind):
        for e in man.get(kind, []):
            R, C = expand(dirty, e["retail"]), expand(clean, e["clean"])
            if not R or not C:
                res.add(kind, f"{e['retail']} | {e['clean']}", 0, True, "missing input (pattern matched nothing)")
                continue
            yield e, R, C

    for e, R, C in pairs("music"):
        from cleanroom.audio.notes import load
        fmt = e.get("format")
        retail = {k: load(p, fmt)[1] for k, p in R.items()}
        for k, p in C.items():
            best, cn = None, load(p, fmt)[1]
            for rk, rn in retail.items():
                r = music.compare(cn, rn, cfg["music"])
                if best is None or r["score"] > best[1]["score"]:
                    best = (rk, r)
            r = best[1]
            res.add("music", k, r["score"], r["fail"],
                    f"vs {best[0]}: ngram {r['ngram']} dtw {r['dtw']} {'; '.join(r['why'])}")
    for e, R, C in pairs("music_pcm"):
        import wave
        for k in sorted(set(R) & set(C)):
            sig = []
            for p in (C[k], R[k]):
                with wave.open(p) as w:
                    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels())
                    sig.append((x.mean(axis=1) / 32768.0, w.getframerate()))
            r = music.compare_pcm(sig[0][0], sig[1][0], sig[0][1], cfg["music"])
            res.add("music_pcm", k, r["chroma_excess"] / cfg["music"]["chroma_excess_max"], r["fail"],
                    f"chroma excess {r['chroma_excess']}")
    names = {}
    if man.get("names"):
        names = json.load(open(os.path.join(base, man["names"]), encoding="utf8"))
    for e, R, C in pairs("text"):
        retail = [s for p in R.values() for s in load_strings(p)]
        for k, p in C.items():
            strings = load_strings(p)
            r = text.check(strings, retail, names, cfg["text"])
            det = "; ".join([f"#{i} shares \"{w}\"" for i, w in r["overlaps"][:3]] +
                            [f"#{i} keeps name {n}" for i, n in r["names"][:3]])
            if r["unmapped_caps"]:
                det += f" (info: unmapped capitalised words {r['unmapped_caps'][:6]})"
            res.add("text", k, len(r["overlaps"]) + len(r["names"]), r["fail"],
                    f"{len(strings)} strings; {len(r['overlaps'])} overlaps, {len(r['names'])} names " + det)
    for e, R, C in pairs("geometry"):
        gcfg = dict(cfg["geometry"], **e.get("thresholds", {}))     # per-entry override (mesh scale differs)
        for k in sorted(set(R) & set(C)):
            r = geometry.compare(geometry.load_mesh(R[k]), geometry.load_mesh(C[k]),
                                 _anchors(e, base, k), gcfg)
            res.add("geometry", k, r.get("score", 0), r["fail"],
                    f"voxel IoU {r['voxel_iou']} silhouettes {r['silhouette']} {'; '.join(r['why'])}")
        for k in sorted(set(R) ^ set(C)):
            res.add("geometry", k, 0, True, "unpaired (exists on one side only)")
    for e, R, C in pairs("tilemap"):
        for k in sorted(set(R) & set(C)):
            g = [json.load(open(p))["grid"] for p in (R[k], C[k])]
            anc = json.load(open(C[k])).get("anchor_cols", [])
            r = tilemap.compare(g[0], g[1], cfg["tilemap"], anc)
            res.add("tilemap", k, r["score"], r["fail"],
                    f"feature IoU {r['feature_iou']} pit Jaccard {r['hole_jaccard']}")
        for k in sorted(set(R) ^ set(C)):
            res.add("tilemap", k, 0, True, "unpaired (exists on one side only)")
    for e, R, C in pairs("images"):
        from cleanroom.gfx import png
        for k in sorted(set(R) & set(C)):
            r = images.compare(_read_img(R[k]), _read_img(C[k]), cfg["images"])
            res.add("images", k, r["score"], r["fail"],
                    "flat" if r["flat"] else f"ssim {r['ssim']} phash {r['phash']} dhash {r['dhash']}")
    for e, R, C in pairs("blockmap"):
        dims = json.load(open(os.path.join(base, e["dims"]))) if isinstance(e.get("dims"), str) else e.get("dims", {})
        for k in sorted(set(R) & set(C)):
            d = dims.get(k)
            if not d:
                res.add("blockmap", k, 0, True, "no dims")
                continue
            w, h = d["w"], d["h"]

            def grid(p):
                a = np.frombuffer(open(p, "rb").read(), np.uint8)[:w * h]
                return np.pad(a, (0, w * h - len(a))).reshape(h, w)
            r = blockmap.compare(grid(R[k]), grid(C[k]), d.get("anchors"), cfg["blockmap"])
            res.add("blockmap", k, r["score"], r["fail"], f"{r['score']} of {r['compared']} free cells equal")
    for e, R, C in pairs("code"):
        def seg(path, a, b):
            data = open(path, "rb").read()
            return data[int(str(a), 0):int(str(b), 0) if b is not None else len(data)]
        rk, ck = next(iter(R)), next(iter(C))
        rc = seg(R[rk], e.get("start", 0), e.get("end"))
        cc = seg(C[ck], e.get("clean_start", e.get("start", 0)), e.get("clean_end", e.get("end")))
        comp = [open(p, "rb").read() for pat in e.get("compiled", []) for p in expand(clean, pat).values()]
        bad = code.check(rc, cc, comp, cfg["code"])
        nbytes = sum(b - a for a, b in bad)
        res.add("code", ck, nbytes, bool(bad),
                f"{len(bad)} retail-only ranges, {nbytes} B" + (f", first at +0x{bad[0][0]:X}" if bad else "")
                + f"; {len(comp)} compiled files")
    return res


def report_md(res, title):
    out = [f"# Originality report: {title}", ""]
    fails = res.failing()
    out.append(f"**{len(fails)} failing** of {sum(len(v) for v in res.rows.values())} checked.")
    out.append("")
    out.append("| kind | checked | failing | worst | worst score |")
    out.append("|---|---|---|---|---|")
    for k, rows in res.rows.items():
        w = max(rows, key=lambda r: r[1])
        out.append(f"| {k} | {len(rows)} | {sum(r[2] for r in rows)} | {w[0]} | {w[1]:.2f} |")
    if fails:
        out += ["", "## Failing", ""]
        out += [f"- {k}: `{r[0]}` {r[3]}" for k, r in fails]
    return "\n".join(out) + "\n"


def find_manifest(spec, explicit=None):
    for p in [explicit, os.path.join(spec, "originality.json"), os.path.join(spec, "..", "originality.json")]:
        if p and os.path.exists(p):
            return p
    return None


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    dirty, clean, spec = argv[1:4]
    mpath = find_manifest(spec, argv[argv.index("--manifest") + 1] if "--manifest" in argv else None)
    if not mpath:
        print(f"originality: no originality.json near {spec}: FAIL")
        return 1
    man = json.load(open(mpath, encoding="utf8"))
    res = run(dirty, clean, man, os.path.dirname(os.path.abspath(mpath)))
    fails = res.failing()
    print(f"originality: {man.get('game', os.path.basename(os.path.dirname(os.path.abspath(mpath))))} "
          f"({mpath}); {len(fails)} failing")
    for k, rows in res.rows.items():
        w = max(rows, key=lambda r: r[1])
        print(f"  {k:10s} {len(rows):5d} checked {sum(r[2] for r in rows):4d} failing   worst {w[1]:.2f} {w[0]}")
    for k, r in fails[:10]:
        print(f"  FAIL {k} {r[0]}: {r[3][:110]}")
    if not res.rows:
        print("  (manifest lists nothing to check)")
    if "--report" in argv:
        rp = argv[argv.index("--report") + 1]
        with open(rp, "w", encoding="utf8") as f:
            f.write(report_md(res, man.get("game", "")))
        print(f"  report -> {rp}")
    return 1 if fails or not res.rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
