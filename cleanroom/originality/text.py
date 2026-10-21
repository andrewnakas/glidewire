"""Text originality: no clean string may share a run of >= run_words words
with any retail string, and no retail proper noun may survive.

Words are lower-cased alphanumeric runs (apostrophes dropped) after control
codes (config control_re) are removed; punctuation and case never matter.
The name map is {"RetailName": "CleanName", ...}; every key is searched for
as a whole word (case-insensitive) in the clean strings.

    words(s, cfg) -> [str]
    shared_runs(clean, retail, cfg) -> [(clean index, "the shared words")]
    name_hits(clean, names) -> [(clean index, retail name)]
    check(clean, retail, names, cfg) -> dict(fail, overlaps, names, unmapped_caps)
"""
import re

from cleanroom.originality.config import CONFIG


def strip_controls(s, cfg=None):
    cfg = cfg or CONFIG["text"]
    return re.sub(cfg["control_re"], " ", s)


def words(s, cfg=None):
    return re.findall(r"[a-z0-9]+", strip_controls(s, cfg).lower().replace("'", "").replace("’", ""))


def _shingles(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def shared_runs(clean, retail, cfg=None):
    cfg = cfg or CONFIG["text"]
    n = cfg["run_words"]
    index = set()
    for s in retail:
        index |= _shingles(words(s, cfg), n)
    out = []
    for i, s in enumerate(clean):
        ws = words(s, cfg)
        hits = [k for k in range(len(ws) - n + 1) if tuple(ws[k:k + n]) in index]
        if hits:
            # report the longest contiguous run
            best, start, run = (0, 0), hits[0], 1
            for a, b in zip(hits, hits[1:] + [None]):
                if b == a + 1:
                    run += 1
                    continue
                if run > best[0]:
                    best = (run, start)
                start, run = b, 1
            out.append((i, " ".join(ws[best[1]:best[1] + best[0] + n - 1])))
    return out


def name_hits(clean, names, cfg=None):
    out = []
    pats = [(k, re.compile(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])", re.I))
            for k in names if k.strip()]
    for i, s in enumerate(clean):
        t = strip_controls(s, cfg)
        for k, p in pats:
            if p.search(t):
                out.append((i, k))
    return out


def unmapped_caps(clean, names, cfg=None):
    """Capitalised words that are not sentence-initial and not a mapped
    clean name or an allowed word: candidates for the name map (info only)."""
    cfg = cfg or CONFIG["text"]
    allowed = {w.lower() for v in names.values() for w in re.findall(r"[A-Za-z]+", v)}
    allowed |= {w.lower() for w in cfg.get("allowed_caps", [])}
    seen = set()
    for s in clean:
        t = strip_controls(s, cfg)
        for m in re.finditer(r"(?<=[a-z0-9,;] )([A-Z][a-z]+)", t):
            w = m.group(1)
            if w.lower() not in allowed:
                seen.add(w)
    return sorted(seen)


def check(clean, retail, names=None, cfg=None):
    cfg = cfg or CONFIG["text"]
    names = names or {}
    ov = shared_runs(clean, retail, cfg)
    nh = name_hits(clean, names, cfg)
    return {"fail": bool(ov or nh), "overlaps": ov, "names": nh,
            "unmapped_caps": unmapped_caps(clean, names, cfg)}
