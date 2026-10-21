"""Music originality: is a clean sequence melodically a copy of a retail one?

Notes come from cleanroom.audio.notes (cseq / m64 / MIDI). Each channel is
reduced to its top line (highest key per onset). Only melodic clean lines
are judged (distinct/total interval 4-grams >= melodic_diversity): drum
grooves, root basses and triad arpeggios are generic figures that any two
pieces share. A copied melody keeps its diversity, so it is always judged;
retail lines are all kept (a melody may be moved to another channel).
Two measures, both transposition-invariant:

  ngram   containment of the clean pitch-interval n-grams (n = 4..6
          intervals) in the pooled retail n-grams, counted over clean
          occurrences; n-grams with a single distinct interval (repeats,
          plain scale runs) are skipped as non-distinctive.
  dtw     best channel-pair similarity of the rhythm-contour (pitch minus
          median, log inter-onset ratio) under banded DTW:
          exp(-mean cost / scale). Pedal and drum lines are skipped.

    compare(clean_notes, retail_notes, cfg) -> dict(ngram={n: x}, dtw, score, fail, why)
                                             score = worst measure / its threshold (> 1 fails)
    compare_pcm(clean, retail, rate, cfg)   -> chroma check for streamed music
"""
import numpy as np

from cleanroom.originality.config import CONFIG


def top_lines(notes, min_notes=8):
    """{channel: (onset ticks, keys)} keeping the highest key per onset."""
    by = {}
    for n in notes:
        line = by.setdefault(n.ch, {})
        line[n.time] = max(n.key, line.get(n.time, -1))
    out = {}
    for ch, line in by.items():
        if len(line) >= min_notes:
            ts = np.array(sorted(line), np.int64)
            out[ch] = (ts, np.array([line[t] for t in ts], np.int64))
    return out


def _ngrams(keys, n):
    iv = np.diff(keys)
    out = []
    for i in range(len(iv) - n + 1):
        g = tuple(int(v) for v in iv[i:i + n])
        if len(set(g)) > 1:
            out.append(g)
    return out


def diversity(keys, n=4):
    """Distinct / total interval 4-grams: low for ostinatos, drums, arps."""
    g = [tuple(keys[i + 1:i + n + 1] - keys[i:i + n]) for i in range(len(keys) - n)]
    return len(set(g)) / len(g) if g else 0.0


def ngram_scores(clean_lines, retail_lines, ns=(4, 5, 6)):
    """Fraction of the distinct clean n-grams found anywhere in retail."""
    res = {}
    for n in ns:
        pool, grams = set(), set()
        for _, k in retail_lines.values():
            pool.update(_ngrams(k, n))
        for _, k in clean_lines.values():
            grams.update(_ngrams(k, n))
        res[str(n)] = round(len(grams & pool) / len(grams), 4) if grams else 0.0
    return res


def contour(ts, keys, limit):
    ts, keys = ts[:limit], keys[:limit].astype(np.float64)
    ioi = np.diff(np.append(ts, ts[-1] + (ts[-1] - ts[-2] if len(ts) > 1 else 1))).astype(np.float64)
    ioi = np.maximum(ioi, 1)
    return keys - np.median(keys), np.log2(ioi / np.median(ioi))


def dtw(a, b, band=0.15, w_rhythm=1.0):
    """Mean cost along the best banded warping path; a, b = (pitch, logioi)."""
    pa, ra = a
    pb, rb = b
    n, m = len(pa), len(pb)
    c = np.abs(pa[:, None] - pb[None, :]) + w_rhythm * np.abs(ra[:, None] - rb[None, :])
    r = max(int(band * max(n, m)), abs(n - m) + 1)
    D = np.full(m, np.inf)
    L = np.zeros(m)
    for i in range(n):
        mid = int(round(i * (m - 1) / max(n - 1, 1)))
        lo, hi = max(0, mid - r), min(m, mid + r + 1)
        ci = c[i, lo:hi]
        jj = np.arange(lo, hi)
        if i == 0:
            a_cost = np.full(hi - lo, np.inf)
            a_cost[0] = 0.0 if lo == 0 else np.inf
            a_len = np.zeros(hi - lo)
        else:
            up, upl = D[lo:hi], L[lo:hi]
            dg = np.concatenate([[np.inf], D[:-1]])[lo:hi]
            dgl = np.concatenate([[0.0], L[:-1]])[lo:hi]
            pick = up <= dg
            a_cost, a_len = np.where(pick, up, dg), np.where(pick, upl, dgl)
        # x[j] = c[j] + min(a[j], x[j-1]) solved with prefix minima
        C = np.cumsum(ci)
        cand = a_cost - (C - ci)
        acc = np.minimum.accumulate(cand)
        k = np.maximum.accumulate(np.where(cand == acc, np.arange(hi - lo), 0))
        D = np.full(m, np.inf)
        L = np.zeros(m)
        D[lo:hi] = C + acc
        L[lo:hi] = a_len[k] + 1 + (np.arange(hi - lo) - k)
    return float(D[-1] / max(L[-1], 1)) if np.isfinite(D[-1]) else float("inf")


def dtw_similarity(clean_lines, retail_lines, cfg):
    best = 0.0
    lim = cfg["dtw_notes"]


    def ok(k, div):
        return np.std(k[:lim]) >= cfg["min_pitch_std"] and diversity(k[:lim]) >= div
    cs = [contour(t, k, lim) for t, k in clean_lines.values() if ok(k, cfg["melodic_diversity"])]
    rs = [contour(t, k, lim) for t, k in retail_lines.values() if ok(k, cfg["min_diversity"])]
    for a in cs:
        for b in rs:
            d = dtw(a, b, cfg["dtw_band"])
            best = max(best, float(np.exp(-d / cfg["dtw_scale"])))
    return round(best, 4)


def compare(clean_notes, retail_notes, cfg=None):
    cfg = cfg or CONFIG["music"]
    cl = top_lines(clean_notes, cfg["min_notes"])
    cl = {c: v for c, v in cl.items() if diversity(v[1][:cfg["dtw_notes"] * 2]) >= cfg["melodic_diversity"]}
    rl = top_lines(retail_notes, cfg["min_notes"])
    ng = ngram_scores(cl, rl, cfg["ns"])
    d = dtw_similarity(cl, rl, cfg)
    why = [f"{n}-gram {v:.2f} > {cfg['ngram_max'][n]}" for n, v in ng.items() if v > cfg["ngram_max"][n]]
    if d > cfg["dtw_sim_max"]:
        why.append(f"contour dtw {d:.2f} > {cfg['dtw_sim_max']}")
    score = max([d / cfg["dtw_sim_max"]] + [v / cfg["ngram_max"][n] for n, v in ng.items()])
    return {"ngram": ng, "dtw": d, "score": round(score, 4), "fail": bool(why), "why": why}


# ----------------------------------------------------------- streamed audio

def chroma(x, rate, frame=4096, hop=2048, max_frames=400):
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    nfr = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(nfr)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(frame), axis=1))
    f = np.fft.rfftfreq(frame, 1 / rate)
    ok = (f >= 60) & (f <= 5000)
    pc = np.round(12 * np.log2(f[ok] / 440.0)).astype(int) % 12
    C = np.zeros((nfr, 12))
    for k in range(12):
        C[:, k] = (spec[:, ok][:, pc == k] ** 2).sum(axis=1)
    if nfr > max_frames:
        g = nfr // max_frames
        C = C[:g * max_frames].reshape(max_frames, g, 12).mean(axis=1)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)


def _path_mean(S, band):
    n, m = S.shape
    cost = 1 - S
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0
    r = max(int(band * max(n, m)), abs(n - m) + 1)
    for i in range(1, n + 1):
        j0, j1 = max(1, int(i * m / n) - r), min(m, int(i * m / n) + r)
        for j in range(j0, j1 + 1):
            opts = ((D[i - 1, j - 1], L[i - 1, j - 1]), (D[i - 1, j], L[i - 1, j]), (D[i, j - 1], L[i, j - 1]))
            dv, lv = min(opts)
            D[i, j], L[i, j] = dv + cost[i - 1, j - 1], lv + 1
    return 1 - D[n, m] / max(L[n, m], 1)


def compare_pcm(clean, retail, rate, cfg=None, band=0.1):
    """Key-invariant chroma similarity of two recordings. excess = mean
    cosine along the best DTW path minus the mean of the whole similarity
    matrix (what any two pieces in the same key share)."""
    cfg = cfg or CONFIG["music"]
    A, B = chroma(clean, rate, max_frames=150), chroma(retail, rate, max_frames=150)
    best = -1.0
    for rot in range(12):
        S = A @ np.roll(B, rot, axis=1).T
        best = max(best, _path_mean(S, band) - float(S.mean()))
    best = round(best, 4)
    lim = cfg["chroma_excess_max"]
    return {"chroma_excess": best, "fail": best > lim,
            "why": [f"chroma excess {best:.2f} > {lim}"] if best > lim else []}
