"""CLEAN ROOM: seeded composer for sequenced music slots.

Input is structure only: tempo, length, loop point, channel count,
instrument roles and mood tags (plus a seed). It never reads retail notes.

    song = compose(bpm=140, length=96*48, loop=16*48, channels=6,
                   roles=["lead", "bass", "chords", "drums"], moods=["dark"], seed=3)
    to_cseq(song) / to_m64(song) / to_midi(song) / to_pcm(song, 32000)
    self_check(song, retail_path)   # dirty-room gate: originality vs one retail seq

    python -m cleanroom.audio.compose --bpm 140 --seconds 60 --loop-seconds 8 \
        --roles lead,bass,chords,drums --mood dark,fast --seed 3 --out song.mid [--check retail.m64]

Roles: lead, counter, bass, chords, pad, arp, drums (drums use keys 36/38/42/46,
MIDI channel 9 when free). Moods: major/bright/heroic/calm/happy pick major
modes, minor/dark/sad/tense/eerie pick minor modes; fast/busy raise note
density, calm/slow lower it, tense adds chromatic approach notes.
Phrases restart at the loop point so the loop seam lands on a phrase start.
"""
import sys
from dataclasses import dataclass, field

import numpy as np

from cleanroom.audio.notes import Note
from cleanroom.decomp.gen import h32

MODES = {"ionian": [0, 2, 4, 5, 7, 9, 11], "lydian": [0, 2, 4, 6, 7, 9, 11],
         "mixolydian": [0, 2, 4, 5, 7, 9, 10], "aeolian": [0, 2, 3, 5, 7, 8, 10],
         "dorian": [0, 2, 3, 5, 7, 9, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
         "harmonic": [0, 2, 3, 5, 7, 8, 11]}
MAJOR_MOODS = {"major", "bright", "heroic", "calm", "happy", "triumphant"}
MINOR_MOODS = {"minor", "dark", "sad", "tense", "eerie", "spooky"}
DEFAULT_ROLES = ["lead", "bass", "chords", "drums", "arp", "counter", "pad"]
DRUM = {"kick": 36, "snare": 38, "hat": 42, "open": 46}


@dataclass
class Song:
    bpm: float
    length: int
    loop: int = None
    division: int = 48
    notes: list = field(default_factory=list)
    roles: dict = field(default_factory=dict)      # channel -> role
    mode: str = ""
    root: int = 60


def _mode(rng, moods):
    m = set(moods)
    if m & MINOR_MOODS and not m & MAJOR_MOODS:
        pool = ["aeolian", "dorian", "phrygian", "harmonic"]
    elif m & MAJOR_MOODS:
        pool = ["ionian", "lydian", "mixolydian"]
    else:
        pool = list(MODES)
    return pool[int(rng.integers(len(pool)))]


def _progression(rng, bars):
    """Random walk over scale degrees (no stock progressions): strong pull to
    the tonic every 4th chord, steps of a 4th/5th favoured."""
    out, d = [], 0
    moves = [3, 4, 5, 1, 6, 2]
    w = np.array([5, 4, 4, 2, 2, 1], float)
    for i in range(bars):
        if i % 4 == 0:
            d = 0 if i % 8 == 0 else int(rng.choice([0, 3, 5]))
        else:
            d = (d + int(rng.choice(moves, p=w / w.sum()))) % 7
        out.append(d)
    return out


def _chord(scale, deg):
    return [scale[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4)]


def _rhythm(rng, beat, density):
    """One bar of onsets/durations (ticks) from cells; density 0..1."""
    cells = [[2], [1, 1], [1.5, 0.5], [0.5, 0.5, 1], [0.5, 0.5, 0.5, 0.5], [1, 0.5, 0.5], [3, 1], [4]]
    weights = np.array([2, 3, 2, 2, 1, 2, 1, 1], float)
    weights[[4, 3, 5]] *= 1 + 3 * density
    weights[[0, 6, 7]] *= 1.5 - density
    out, t = [], 0.0
    while t < 4:
        cell = cells[int(rng.choice(len(cells), p=weights / weights.sum()))]
        for d in cell:
            if t + d > 4:
                d = 4 - t
            out.append((int(t * beat), max(1, int(d * beat))))
            t += d
            if t >= 4:
                break
    return out


def _motif(rng, scale, beat, density, span):
    rh = _rhythm(rng, beat, density)
    steps = rng.choice([-2, -1, -1, 1, 1, 2, 3, -3, 0, 4], size=len(rh))
    pos, line = int(rng.integers(0, 5)), []
    for st in steps:
        pos = int(np.clip(pos + st, -span, span + 4))
        line.append(pos)
    return rh, line


def _vary(rng, line, kind):
    if kind == 1:
        return [line[0] - (p - line[0]) for p in line]               # inversion
    if kind == 2:
        return [p + int(rng.choice([-2, 2, 3])) for p in line]       # sequence
    if kind == 3:
        out = list(line)
        for i in rng.choice(len(out), size=max(1, len(out) // 3), replace=False):
            out[i] += int(rng.choice([-1, 1, 2]))
        return out
    return line


def _deg_key(root, scale, pos):
    return root + scale[pos % 7] + 12 * (pos // 7)


def compose(bpm=120, length=None, loop=None, channels=None, roles=None, moods=(), seed=0,
            division=48, seconds=None, loop_seconds=None) -> Song:
    """Length/loop in ticks, or seconds via seconds=/loop_seconds=."""
    tick_s = 60.0 / bpm / division
    if seconds is not None:
        length = int(round(seconds / tick_s))
    if loop_seconds is not None:
        loop = int(round(loop_seconds / tick_s))
    length = length or division * 4 * 16
    channels = channels or (len(roles) if roles else 4)
    roles = list(roles or DEFAULT_ROLES[:channels])[:channels]
    moods = [m.lower() for m in moods]
    rng = np.random.default_rng(h32("compose", seed, bpm, length, loop, ",".join(roles), ",".join(moods)))
    mode = _mode(rng, moods)
    scale = MODES[mode]
    root = 48 + int(rng.integers(0, 12))
    density = 0.5 + 0.35 * bool({"fast", "busy", "action"} & set(moods)) - 0.35 * bool({"calm", "slow"} & set(moods))
    tense = "tense" in moods
    beat, bar = division, division * 4
    # channel assignment: drums prefer MIDI channel 9
    chans, free = {}, [c for c in range(16) if c != 9]
    for r in roles:
        chans[r if r not in chans else f"{r}{len(chans)}"] = 9 if r == "drums" and 9 not in chans.values() else free.pop(0)
    song = Song(bpm, length, loop, division, [], {c: r.rstrip("0123456789") for r, c in chans.items()}, mode, root)
    sections = [(0, loop)] if loop else []
    sections.append((loop or 0, length))
    for s0, s1 in sections:
        if s1 <= s0:
            continue
        nbars = -(-(s1 - s0) // bar)
        prog = _progression(rng, nbars)
        motifs = [_motif(rng, scale, beat, density, 5) for _ in range(2)]
        form = [0, 0, 1, 0] if rng.uniform() < 0.5 else [0, 1, 0, 1]
        for b in range(nbars):
            t0 = s0 + b * bar
            deg = prog[b]
            chord = _chord(scale, deg)
            phrase, pos_in = form[(b // 2) % 4], b % 2
            for name, ch in chans.items():
                role = name.rstrip("0123456789")
                ev = _bar(rng, role, t0, beat, bar, root, scale, deg, chord, motifs[phrase], phrase,
                          pos_in, density, tense, b)
                for (t, k, v, d) in ev:
                    if t < s1:
                        song.notes.append(Note(t, ch, int(np.clip(k, 0, 127)), v, max(1, min(d, s1 - t))))
    song.notes.sort()
    return song


def _bar(rng, role, t0, beat, bar, root, scale, deg, chord, motif, phrase, pos_in, density, tense, b):
    ev = []
    if role == "drums":
        for q in range(4):
            if q in (0, 2) or (density > 0.6 and q == 3 and b % 2):
                ev.append((t0 + q * beat, DRUM["kick"], 110, beat // 2))
            if q in (1, 3):
                ev.append((t0 + q * beat, DRUM["snare"], 100, beat // 2))
            for h in range(2 if density < 0.7 else 4):
                step = beat // (2 if density < 0.7 else 4)
                ev.append((t0 + q * beat + h * step, DRUM["hat"] if (q + h) % 4 else DRUM["open"], 70, step))
    elif role == "bass":
        pattern = rng.choice([0, 0, 4, 7, 0, 2]) if density > 0.6 else 0
        steps = 8 if density > 0.6 else 4
        for k in range(steps):
            d = bar // steps
            off = 0 if k % 2 == 0 else int(pattern)
            ev.append((t0 + k * d, root - 12 + chord[0] + (scale[off % 7] if off else 0), 96, d - 2))
    elif role in ("chords", "pad"):
        hits = [0] if role == "pad" or density < 0.4 else [0, 2 * beat]
        for h in hits:
            for k in chord:
                ev.append((t0 + h, root + k, 72, (bar if role == "pad" else 2 * beat) - 4))
    elif role == "arp":
        n = 8 if density < 0.7 else 16
        order = [0, 1, 2, 1] if b % 2 else [0, 1, 2, 3]
        for k in range(n):
            tones = chord + [chord[0] + 12]
            ev.append((t0 + k * bar // n, root + 12 + tones[order[k % 4]], 64, bar // n - 2))
    elif role in ("lead", "counter"):
        rh, line = motif
        line = _vary(rng, line, (phrase * 2 + pos_in + (b // 4)) % 4)
        shift = deg if role == "lead" else deg - 2
        base = root + (12 if role == "lead" else 0)
        rest_p = 0.1 if role == "lead" else 0.35
        for (t, d), pos in zip(rh, line):
            if rng.uniform() < rest_p:
                continue
            k = _deg_key(base, scale, pos + shift)
            if tense and rng.uniform() < 0.15:
                k += int(rng.choice([-1, 1]))
            ev.append((t0 + t, k, 100 if role == "lead" else 80, max(1, int(d * 0.9))))
    return ev


# ---------------------------------------------------------------- writers

def to_cseq(song, programs=None):
    from cleanroom.audio import cseq
    tracks = {}
    usec = int(round(60e6 / song.bpm))
    for ch, role in song.roles.items():
        ev = [(0, "midi", 0xC0 | ch, (programs or {}).get(role, 0) & 0x7F)]
        if ch == min(song.roles):
            ev.append((0, "tempo", usec))
        if song.loop is not None:
            ev.append((song.loop, "loopstart"))
        ev += [(n.time, "note", n.ch, n.key, n.vel, n.dur) for n in song.notes if n.ch == ch]
        if song.loop is not None:
            ev.append((song.length, "loopend", 0xFF))
        tracks[ch] = ev
    return cseq.encode(tracks, song.division)


def to_m64(song, programs=None):
    from cleanroom.audio import m64
    scale = 48 / song.division
    notes = [Note(int(n.time * scale), n.ch, n.key, n.vel, max(1, int(n.dur * scale))) for n in song.notes]
    progs = {ch: (programs or {}).get(r, 0) for ch, r in song.roles.items()}
    return m64.write(notes, song.bpm, int(song.length * scale),
                     None if song.loop is None else int(song.loop * scale), progs)


def to_midi(song, programs=None):
    from cleanroom.audio import midi
    progs = {ch: (programs or {}).get(r, 0) for ch, r in song.roles.items()}
    return midi.write(song.notes, song.division, int(round(60e6 / song.bpm)),
                      (song.loop, song.length) if song.loop is not None else None, progs)


def to_pcm(song, rate=32000):
    """Preview / streamed-slot render with simple oscillators (float32 mono)."""
    tick = 60.0 / song.bpm / song.division
    out = np.zeros(int(song.length * tick * rate) + rate, np.float64)
    rng = np.random.default_rng(1)
    for n in song.notes:
        role = song.roles.get(n.ch, "lead")
        at, ln = int(n.time * tick * rate), max(16, int(n.dur * tick * rate))
        t = np.arange(ln) / rate
        if role == "drums":
            ln = int(0.2 * rate)
            t = np.arange(ln) / rate
            x = (np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-t * 30)) / rate) if n.key == DRUM["kick"]
                 else rng.uniform(-1, 1, ln) * (0.6 if n.key == DRUM["snare"] else 0.25))
            x = x * np.exp(-t * (10 if n.key == DRUM["kick"] else 25))
        else:
            f = 440.0 * 2 ** ((n.key - 69) / 12)
            ph = f * t
            x = {"bass": np.sin(2 * np.pi * ph) + 0.3 * np.sin(4 * np.pi * ph),
                 "lead": 2 * (ph % 1) - 1, "arp": np.where(ph % 1 < 0.3, 1.0, -1.0)}.get(role, np.sin(2 * np.pi * ph))
            x = x * np.minimum(1, t / 0.005) * np.exp(-t * 2.0)
        m = min(len(x), len(out) - at)
        out[at:at + m] += x[:m] * n.vel / 127 * 0.25
    out = np.tanh(out[:int(song.length * tick * rate)])
    return out.astype(np.float32)


def self_check(song, retail, cfg=None):
    """Dirty-room gate: originality.music.compare of this song against a
    retail sequence (path or Note list). The composer itself never sees it."""
    from cleanroom.originality import music
    if isinstance(retail, str):
        from cleanroom.audio.notes import load
        retail = load(retail)[1]
    return music.compare(song.notes, retail, cfg)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m cleanroom.audio.compose")
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--loop", type=int)
    ap.add_argument("--loop-seconds", type=float)
    ap.add_argument("--channels", type=int)
    ap.add_argument("--roles", default="")
    ap.add_argument("--mood", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help=".mid / .m64 / .seq (cseq) / .wav")
    ap.add_argument("--check", help="retail sequence to run the originality check against")
    a = ap.parse_args(argv)
    song = compose(a.bpm, a.ticks, a.loop, a.channels, [r for r in a.roles.split(",") if r] or None,
                   [m for m in a.mood.split(",") if m], a.seed, seconds=a.seconds, loop_seconds=a.loop_seconds)
    ext = a.out.rsplit(".", 1)[-1].lower()
    if ext == "wav":
        import wave
        pcm = (to_pcm(song) * 32767).astype("<i2").tobytes()
        with wave.open(a.out, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(32000), w.writeframes(pcm)
        data = pcm
    else:
        data = {"mid": to_midi, "midi": to_midi, "m64": to_m64}.get(ext, to_cseq)(song)
        open(a.out, "wb").write(data)
    print(f"compose: {a.out}  {len(data)} B  {len(song.notes)} notes  {song.mode} root {song.root}  "
          f"{song.length} ticks loop {song.loop}  roles {sorted(set(song.roles.values()))}")
    if a.check:
        r = self_check(song, a.check)
        print(f"  originality vs {a.check}: ngram {r['ngram']} dtw {r['dtw']} -> {'FAIL ' + '; '.join(r['why']) if r['fail'] else 'ok'}")
        return 1 if r["fail"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
