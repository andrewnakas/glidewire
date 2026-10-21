"""Our own music for the streamed BGM slots: a small procedural rock/techno composer.

    music.compose(name, d) -> float waveform (length d['n'] at d['rate'])

Nothing is derived from the retail recordings except the kept facts: slot length and rate. Where the
arrangement thins out or builds up is our own seeded section plan (2026-10 originality rebuild: the retail
loudness curve is no longer used). Each slot gets its own seeded key, tempo, progression, riff and lead motif.
"""
import numpy as np

from cleanroom.decomp.gen import h32

MINOR = [0, 2, 3, 5, 7, 8, 10]
PROGS = [[0, 5, 2, 6], [0, 3, 5, 4], [0, 6, 5, 6], [0, 5, 3, 4], [0, 2, 5, 4], [0, 6, 3, 4]]


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def saw(ph):
    return 2.0 * (ph - np.floor(ph + 0.5))


def square(ph, duty=0.5):
    return np.where((ph % 1.0) < duty, 1.0, -1.0)


def env_ad(n, rate, a=0.004, d=0.25):
    t = np.arange(n) / rate
    return np.minimum(1.0, t / max(a, 1e-4)) * np.exp(-t / d)


def lowpass(x, cutoff, rate):
    a = np.exp(-2 * np.pi * cutoff / rate)
    y = np.empty_like(x)
    acc = 0.0
    # one-pole filter, vectorised in blocks via lfilter
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)


def kick(rate):
    n = int(0.28 * rate)
    t = np.arange(n) / rate
    f = 50 + 110 * np.exp(-t * 28)
    ph = np.cumsum(f) / rate
    return np.sin(2 * np.pi * ph) * np.exp(-t * 9)


def snare(rate, rng):
    n = int(0.22 * rate)
    t = np.arange(n) / rate
    noise = rng.uniform(-1, 1, n)
    from scipy.signal import lfilter
    noise = lfilter([1, -1], [1, -0.6], noise)
    return (0.55 * noise * np.exp(-t * 18) + 0.45 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25))


def hat(rate, rng, open_=False):
    n = int((0.18 if open_ else 0.05) * rate)
    t = np.arange(n) / rate
    from scipy.signal import lfilter
    x = lfilter([1, -1], [1, 0.2], rng.uniform(-1, 1, n))
    return x * np.exp(-t * (14 if open_ else 70)) * 0.35


def place(buf, snd, at, gain=1.0):
    if at >= len(buf):
        return
    m = min(len(snd), len(buf) - at)
    buf[at:at + m] += snd[:m] * gain


def tone(freq, n, rate, kind, rng, detune=0.0):
    t = np.arange(n) / rate
    ph = freq * t + rng.uniform()
    if kind == "saw":
        x = saw(ph) + 0.6 * saw(ph * (1 + detune))
    elif kind == "square":
        x = square(ph, 0.3 + 0.2 * rng.uniform())
    elif kind == "dist":
        x = np.tanh(3.5 * (saw(ph) + saw(ph * 1.5 * (1 + detune)) * 0.7))   # power chord, overdriven
    else:
        x = np.sin(2 * np.pi * ph)
    return x


def compose(name, d):
    rate, n = d["rate"], d["n"]
    rng = np.random.default_rng(h32("fzxmusic", name))
    style = rng.integers(0, 3)           # 0 rock, 1 techno, 2 heavy
    bpm = float(rng.choice([132, 140, 146, 150, 156, 162, 170]))
    root = int(rng.integers(40, 48))     # E2..B2
    prog = PROGS[int(rng.integers(len(PROGS)))]
    beat = int(rate * 60 / bpm)
    bar = beat * 4
    out = np.zeros(n + bar, np.float64)
    drums = np.zeros_like(out)
    bass = np.zeros_like(out)
    gtr = np.zeros_like(out)
    lead = np.zeros_like(out)
    K, S, H, HO = kick(rate), snare(rate, rng), hat(rate, rng), hat(rate, rng, True)
    # our own arrangement curve per second (0..1): quiet intro, then seeded sections of build / full / break
    secs = n // rate + 1
    en = np.empty(secs)
    intro = int(rng.integers(4, 10))
    en[:intro] = 0.3
    t = intro
    while t < secs:
        span = int(rng.integers(8, 20))
        en[t:t + span] = float(rng.choice([0.45, 0.7, 0.9, 1.0], p=[0.2, 0.3, 0.3, 0.2]))
        t += span

    def energy_at(sample):
        return float(en[min(len(en) - 1, sample // rate)])

    motif = [int(v) for v in rng.choice([0, 2, 3, 4, 6, 7, 9], size=8)]
    rhythm = rng.choice([0.5, 0.5, 1.0, 0.25], size=8)
    riff = [int(v) for v in rng.choice([0, 0, 0, 7, 10, 12, 5], size=8)]
    nbars = n // bar + 1
    for b in range(nbars):
        s0 = b * bar
        e = energy_at(s0)
        deg = prog[(b // 2) % len(prog)]
        chord_root = root + MINOR[deg % 7] + (12 if deg >= 7 else 0)
        section = (b // 8) % 4                      # verse / build / chorus / break
        full = e > 0.55 and section != 3
        # drums
        for q in range(4):
            at = s0 + q * beat
            if style == 1 or q in (0, 2) or (full and q == 3 and b % 2):
                place(drums, K, at, 0.9)
            if q in (1, 3) and e > 0.25:
                place(drums, S, at, 0.6)
            for h in range(2):
                place(drums, HO if (style == 1 and h == 1) else H, at + h * beat // 2, 0.5 if h else 0.35)
        # bass: eighth notes following the riff
        for k in range(8):
            f = midi_hz(chord_root - 12 + riff[k] % 12)
            m = beat // 2
            x = tone(f, m, rate, "saw", rng, 0.004) * env_ad(m, rate, 0.003, 0.18)
            place(bass, x, s0 + k * m, 0.45)
        # rhythm guitar (power chords) on the fuller sections
        if style != 1 and e > 0.35:
            for k in range(2 if not full else 8):
                m = bar // (2 if not full else 8)
                x = tone(midi_hz(chord_root), m, rate, "dist", rng, 0.002) * env_ad(m, rate, 0.005, 0.5)
                place(gtr, x, s0 + k * m, 0.22)
        # lead: seeded motif over the chord, only in chorus/build sections
        if section in (1, 2) and e > 0.4:
            at = s0
            for k in range(8):
                dur = int(beat * rhythm[k])
                deg2 = (deg + motif[k]) % 7
                f = midi_hz(root + 24 + MINOR[deg2])
                x = tone(f, dur, rate, "square" if style != 2 else "saw", rng, 0.006) * env_ad(dur, rate, 0.01, 0.35)
                place(lead, x, at, 0.2)
                at += dur
                if at >= s0 + bar:
                    break
        elif style == 1 and e > 0.3:
            for k in range(16):                     # arpeggio
                m = beat // 4
                f = midi_hz(chord_root + 12 + [0, 3, 7, 12][k % 4])
                place(lead, tone(f, m, rate, "square", rng) * env_ad(m, rate, 0.002, 0.08), s0 + k * m, 0.12)
    gtr = lowpass(gtr, 3200, rate)
    lead = lowpass(lead, 5000, rate)
    bass = lowpass(bass, 900, rate)
    out = drums + bass + gtr + lead
    # follow the kept loudness curve gently
    curve = np.interp(np.arange(len(out)) / rate, np.arange(len(en)) + 0.5, 0.55 + 0.45 * en)
    out *= curve
    out = np.tanh(out * 0.9)
    x = out[:n]
    fade = min(len(x) // 20, rate // 4)
    if fade > 0:
        x[-fade:] *= np.linspace(1, 0, fade)
        x[:64] *= np.linspace(0, 1, 64)
    return x
