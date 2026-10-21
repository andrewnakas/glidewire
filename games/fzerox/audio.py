"""F-Zero X audio: dirty-room facts and clean-room samples.

    python -m games.fzerox.audio spec <dirty tree>     # DIRTY: facts -> games/fzerox/spec/audio.json
    python -m games.fzerox.audio gen <clean tree>      # CLEAN: bin/us/rev0/audio_table.bin + audio_bank.c

Layout (decomp src/audio/rom/audio_tables.c, kept code): two soundfonts. Font 0 (sound effects, 71
instruments) plays MEDIUM_RAM samples from sample bank 0 (SFX) and MEDIUM_LBA samples from bank 1 (BGM:
the streamed music tracks and the announcer). Font 1 (guitar) uses bank 2. Every sample is ADPCM with a
2-predictor order-2 book; no loop carries a predictor state.

Kept facts (structure only): per sample its length, rate (from the instrument tuning), loop flag, RMS, median
pitch and one tonal-vs-noisy number. The soundfont C (instruments, envelopes, tunings, sample headers) is kept
structure; every AdpcmBook's coefficients are replaced with our own (same order/predictor count, so sizes stay).
Generated: music tracks by our own procedural composer (music.py), sound effects designed by our recipes
(sfx.py; no retail spectra), announcer lines in our own words by Piper TTS (voices.py, voice_lines.json).

    python -m games.fzerox.audio gate <dirty tree> <clean tree> <gate dir>   # music WAVs for the PCM check
"""
import json
import os
import re
import sys

import numpy as np

from cleanroom.audio import descriptor, vadpcm
from cleanroom.audio.pitch import median_f0
from cleanroom.decomp.gen import h32

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec")
VOICES = os.path.join(HERE, "voices")
VER = "us/rev0"
BANK_C = f"src/assets/{VER}/audio_bank/audio_bank.c"
OUT_RATE = 32000

NUM = r"(0x[0-9A-Fa-f]+|-?\d+)"
SAMPLE_RX = re.compile(r"/\* Reference Offset (0x[0-9A-F]+) \*/\s*Sample (\w+) = \{\s*CODEC_ADPCM, (\w+), (\w+), (\w+), "
                       + NUM + r",\s*" + NUM + r", " + NUM + r", " + NUM)
BOOK_RX = re.compile(r"/\* Reference Offset (0x[0-9A-F]+) \*/\s*AdpcmBook (\w+) = \{\s*\{ (\d+), (\d+) \},([^;]*?)\};")
TUNED_RX = re.compile(r"\{ (0x[0-9A-Fa-f]+), ([0-9.]+)f \}")


def s16(v):
    v = int(v, 0)
    return v - 0x10000 if v >= 0x8000 else v


def parse_bank(src):
    """-> samples {name: {...}}, books {ref: (name, order, npred, coefs)} ; refs are per-font offsets."""
    fonts = [(m.start(), m.group(1)) for m in re.finditer(r"u32 (aAudioSoundFont\w+?)Offsets\[\]", src)]

    def font_of(pos):
        f = None
        for start, name in fonts:
            if start <= pos:
                f = name
        return f

    books = {}
    for m in BOOK_RX.finditer(src):
        coefs = [s16(v) for v in re.findall(r"0x[0-9A-Fa-f]+|-?\d+", m.group(5))]
        books[(font_of(m.start()), int(m.group(1), 16))] = (m.group(2), int(m.group(3)), int(m.group(4)), coefs)
    tunings = {}
    for m in re.finditer(r"/\* Reference Offset (0x[0-9A-F]+) \*/\s*(?:Instrument|Drum) (\w+) = \{([^;]*)\};", src):
        for off, t in TUNED_RX.findall(m.group(3)):
            if int(off, 16):
                tunings.setdefault((font_of(m.start()), int(off, 16)), set()).add(float(t))
    samples = {}
    for m in SAMPLE_RX.finditer(src):
        font = font_of(m.start())
        ref = int(m.group(1), 16)
        medium = m.group(3)
        bank = {("aAudioSoundFontSE", "MEDIUM_RAM"): 0, ("aAudioSoundFontSE", "MEDIUM_LBA"): 1,
                ("aAudioSoundFontGuitar", "MEDIUM_RAM"): 2}[(font, medium)]
        t = sorted(tunings.get((font, ref), {1.0}))
        samples[m.group(2)] = {"font": font, "ref": ref, "bank": bank, "medium": medium,
                               "size": int(m.group(6), 0), "addr": int(m.group(7), 0),
                               "loop": int(m.group(8), 0), "book": int(m.group(9), 0),
                               "tunings": t, "rate": int(round(t[-1] * OUT_RATE))}
    return samples, books


def bank_bases(tree):
    src = open(os.path.join(tree, "src/audio/rom/audio_tables.c")).read()
    part = src[src.index("gSampleBankTableData"):]
    return [(int(a, 16), int(b, 16)) for a, b in
            re.findall(r"\{\s*(0x[0-9A-Fa-f]+),\s*(0x[0-9A-Fa-f]+),\s*MEDIUM_CART", part)][:3]


def kind_of(d):
    secs = d["n"] / d["rate"]
    if d["bank"] == 1 and secs > 8:
        return "music"
    if d["bank"] == 1 and secs > 4.5:
        return "jingle"
    if d["bank"] == 2:
        return "guitar"
    return "sfx"


def spec(dirty):
    src = open(os.path.join(dirty, BANK_C)).read()
    samples, books = parse_bank(src)
    bases = bank_bases(dirty)
    table = open(os.path.join(dirty, "bin", VER, "audio_table.bin"), "rb").read()
    out = {}
    for name, s in samples.items():
        _, order, npred, coefs = books[(s["font"], s["book"])]
        start = bases[s["bank"]][0] + s["addr"]
        data = table[start:start + s["size"]]
        n = s["size"] // 9 * 16
        pcm = vadpcm.decode(data, {"order": order, "npred": npred, "book": coefs}, n)
        x = pcm.astype(np.float64) / 32768.0
        fr = descriptor.describe(pcm, s["rate"]).get("frames") or []
        hs = [f.get("h", 0) for f in fr if f.get("h") is not None]
        # kept: one tonal-vs-noisy number and the median pitch (no spectra, no loudness curves)
        d = dict(s, n=n, order=order, npred=npred, harm=round(float(np.mean(hs)), 2) if hs else 0.0,
                 f0=median_f0(x, s["rate"]), rms=float(np.sqrt(np.mean(x ** 2))))
        d["kind"] = kind_of(d)
        out[name] = d
    os.makedirs(SPEC, exist_ok=True)
    json.dump({"bases": bases, "table_size": len(table), "samples": out}, open(os.path.join(SPEC, "audio.json"), "w"))
    kinds = {}
    for d in out.values():
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"samples: {len(out)} {kinds}; books {len(books)}; table {len(table)} B; banks {bases}")


def k_predictors(x, k):
    """Our own order-2 predictor set with k entries: k-means over per-frame least-squares fits."""
    fits = []
    step = max(16, (len(x) // 20000) // 16 * 16)
    for s in range(2, len(x) - 16, step):
        y, p1, p2 = x[s:s + 16], x[s - 1:s + 15], x[s - 2:s + 14]
        if (y ** 2).sum() < 1e3:
            continue
        a, *_ = np.linalg.lstsq(np.stack([p1, p2], 1), y, rcond=None)
        fits.append(a)
    base = [(0.0, 0.0), (1.0, 0.0), (1.8, -0.82), (1.95, -0.96), (1.5, -0.6), (0.5, 0.0), (1.9, -0.92), (1.2, -0.3)]
    if len(fits) < k:
        return base[:k]
    f = np.clip(np.asarray(fits), [-1.95, -0.98], [1.95, 0.98])
    order = np.argsort(f[:, 0])
    c = np.stack([f[order[int((i + 0.5) * len(f) / k)]] for i in range(k)])
    for _ in range(12):
        lab = np.argmin(((f[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = f[lab == j].mean(0)
    out = []
    for a1, a2 in c:
        a2 = float(np.clip(a2, -0.98, 0.98))
        out.append((float(np.clip(a1, -(1 - a2) + 0.02, (1 - a2) - 0.02)), a2))
    return out


def waveform(name, d):
    n, rate = d["n"], d["rate"]
    vp = os.path.join(VOICES, name + ".wav")
    if os.path.exists(vp):
        from .voices import read_wav
        x = read_wav(vp, rate)
        x = np.concatenate([x, np.zeros(max(0, n - len(x)))])[:n]
    elif d["kind"] in ("music", "jingle"):
        from . import music
        x = music.compose(name, d)
    else:
        from . import sfx
        x, _ = sfx.synth(name, d)
    x = np.asarray(x, np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms > 1e-6 and d.get("rms"):
        x = x * (d["rms"] / rms)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0.98:
        x = x * (0.98 / peak)
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int64)


def _one(item):
    name, d = item
    pcm = waveform(name, d)
    preds = k_predictors(pcm.astype(np.float64), d["npred"])
    book = vadpcm.make_book(preds)
    while max(abs(v) for v in book["book"]) > 32767:
        preds = [(a1 * 0.97, a2 * 0.97) for a1, a2 in preds]
        book = vadpcm.make_book(preds)
    data, book, _ = vadpcm.encode(pcm, book)
    data = (data + bytes(d["size"]))[:d["size"]]
    return name, data, [int(np.clip(v, -32768, 32767)) for v in book["book"]]


def fmt_book(coefs):
    lines = []
    for i in range(0, len(coefs), 6):
        lines.append("    " + ", ".join(f"0x{v & 0xFFFF:04X}" for v in coefs[i:i + 6]) + ",")
    return "\n" + "\n".join(lines) + "\n"


def gen(clean, jobs=None, only=None):
    from multiprocessing import Pool
    J = json.load(open(os.path.join(SPEC, "audio.json")))
    S, bases = J["samples"], J["bases"]
    table = bytearray(J["table_size"])
    newbooks = {}
    items = sorted(S.items(), key=lambda kv: -kv[1]["n"])
    if only:
        # re-encode just these samples; keep the rest of the clean table and books
        table = bytearray(open(os.path.join(clean, "bin", VER, "audio_table.bin"), "rb").read())
        _, cur = parse_bank(open(os.path.join(clean, BANK_C)).read())
        newbooks = {k: v[3] for k, v in cur.items()}
        items = [kv for kv in items if kv[0] in only]
    with Pool(jobs or 4) as pool:
        for i, (name, data, coefs) in enumerate(pool.imap_unordered(_one, items)):
            d = S[name]
            start = bases[d["bank"]][0] + d["addr"]
            table[start:start + d["size"]] = data
            newbooks[(d["font"], d["book"])] = coefs
            if i % 10 == 0:
                print(f"  {i}/{len(S)}", flush=True)
    os.makedirs(os.path.join(clean, "bin", VER), exist_ok=True)
    open(os.path.join(clean, "bin", VER, "audio_table.bin"), "wb").write(table)
    # audio_bank.c: the kept structure (from spec/audio_bank_layout.c) with our coefficients
    src = open(os.path.join(SPEC, "audio_bank_layout.c")).read()
    fonts = [(m.start(), m.group(1)) for m in re.finditer(r"u32 (aAudioSoundFont\w+?)Offsets\[\]", src)]

    def rep(m):
        font = [n for s, n in fonts if s <= m.start()][-1]
        coefs = newbooks.get((font, int(m.group(1), 16)))
        if coefs is None:
            coefs = [0] * (int(m.group(3)) * int(m.group(4)) * 8)
        return m.group(0)[:m.start(5) - m.start(0)] + fmt_book(coefs) + "};"

    out = BOOK_RX.sub(rep, src)
    p = os.path.join(clean, BANK_C)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", newline="\n").write(out)
    print(f"audio: {len(S)} samples generated; {len(newbooks)} books; table {len(table)} B")


def layout(dirty):
    """DIRTY: audio_bank.c with every codebook coefficient zeroed -> spec/audio_bank_layout.c (structure only)."""
    src = open(os.path.join(dirty, BANK_C)).read()

    def rep(m):
        return m.group(0)[:m.start(5) - m.start(0)] + fmt_book([0] * (int(m.group(3)) * int(m.group(4)) * 8)) + "};"

    out, n = BOOK_RX.subn(rep, src)
    open(os.path.join(SPEC, "audio_bank_layout.c"), "w", newline="\n").write(out)
    print(f"layout: {n} books zeroed")


def gate(dirty, clean, out, step=4):
    """Music slots decoded from the dirty and the clean tree -> <out>/{retail,clean}_music/<name>.wav (mono,
    downsampled by `step`) for the originality PCM check. The retail WAVs stay on the dirty side."""
    import wave
    for side, tree, sub in (("retail", dirty, "retail_music"), ("clean", clean, "clean_music")):
        src = open(os.path.join(tree, BANK_C)).read()
        samples, books = parse_bank(src)
        bases = bank_bases(tree)
        table = open(os.path.join(tree, "bin", VER, "audio_table.bin"), "rb").read()
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        n_out = 0
        for name, s in samples.items():
            if s["bank"] != 1:
                continue
            _, order, npred, coefs = books[(s["font"], s["book"])]
            start = bases[s["bank"]][0] + s["addr"]
            n = s["size"] // 9 * 16
            if n / s["rate"] < 4.5:
                continue            # spoken lines and stings, not music
            pcm = vadpcm.decode(table[start:start + s["size"]], {"order": order, "npred": npred, "book": coefs}, n)
            x = np.asarray(pcm, np.int16)[::step]
            with wave.open(os.path.join(out, sub, name + ".wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(s["rate"] // step)
                w.writeframes(x.tobytes())
            n_out += 1
        print(f"audio gate: {side} {n_out} music slots -> {out}/{sub}")


if __name__ == "__main__":
    cmd, tree = sys.argv[1], sys.argv[2]
    if cmd == "gate":
        gate(tree, sys.argv[3], sys.argv[4])
        sys.exit(0)
    if cmd == "spec":
        spec(tree)
        layout(tree)
    elif cmd == "voices":     # re-encode only the spoken samples (games/fzerox/voices/*.wav)
        gen(tree, only=[f[:-4] for f in os.listdir(VOICES) if f.endswith(".wav")])
    else:
        gen(tree, only=sys.argv[3:] or None)
