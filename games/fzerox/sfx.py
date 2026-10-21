"""CLEAN ROOM: sound effects designed from scratch (no retail spectra).

Kept per slot (structure only): length, playback rate, whether it loops, loudness, a median pitch when the
slot is tonal, and the slot's mean harmonicity as one number (tonal vs noisy). From those our own recipes
pick a sound: engine drones, blips, chirps, hits and whooshes, seeded by the slot name so every slot sounds
different. Nothing follows the retail spectral outline.

    synth(name, d) -> float waveform in [-1, 1], len d["n"], at d["rate"]
"""
import numpy as np

from cleanroom.decomp.gen import h32


def _harm(d):
    if "harm" in d:
        return float(d["harm"])
    fr = (d.get("desc") or {}).get("frames") or []
    hs = [f.get("h", 0.0) for f in fr if f.get("h") is not None]
    return float(np.mean(hs)) if hs else 0.0


def _lp(x, a):
    """one-pole low-pass, a in (0, 1]: higher = brighter"""
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def _lp_fast(x, a):
    from scipy.signal import lfilter
    return lfilter([a], [1, -(1 - a)], x)


def _band(x, rate, lo, hi):
    from scipy.signal import butter, sosfilt
    lo = max(20.0, min(lo, rate * 0.45))
    hi = max(lo * 1.2, min(hi, rate * 0.48))
    return sosfilt(butter(2, [lo, hi], btype="band", fs=rate, output="sos"), x)


def _env(n, attack, decay_power):
    t = np.linspace(0, 1, n, endpoint=False)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * (1 - t) ** decay_power


def synth(name, d):
    n, rate = int(d["n"]), int(d["rate"])
    rng = np.random.default_rng(h32("glidewire-sfx", name))
    t = np.arange(n) / rate
    secs = n / rate
    tonal = _harm(d) > 0.35
    f0 = d.get("f0") or 0
    looped = bool(d.get("loop")) and secs > 0.25
    base = float(f0) if f0 and 40 < float(f0) < 2000 else float(rng.choice([110, 147, 196, 262, 330]))
    if looped and tonal:
        # engine / hum drone: detuned saw stack + sub, slow wobble, darkened
        kind = "drone"
        wob = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(3, 7) * t)
        x = np.zeros(n)
        for k, det in enumerate((1.0, 1.007, 0.995)):
            ph = np.cumsum(base * det * wob / rate)
            x += 2 * (ph % 1.0) - 1
        x += 0.8 * np.sin(2 * np.pi * np.cumsum(base * 0.5 * wob / rate))
        x = _lp_fast(x, rng.uniform(0.08, 0.2))
    elif looped:
        # wind / rush loop: band noise with a slow sweep
        kind = "rush"
        x = rng.standard_normal(n)
        c = rng.uniform(400, 2500)
        x = _band(x, rate, c * 0.6, c * 1.6)
        x *= 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t)
    elif tonal and secs < 0.45:
        # blip: square/sine chirp with a quick decay
        kind = "blip"
        glide = rng.choice([1.0, 1.5, 0.75, 2.0])
        f = base * (1 + (glide - 1) * t / secs)
        ph = np.cumsum(f / rate)
        x = np.sign(np.sin(2 * np.pi * ph)) * 0.5 + np.sin(2 * np.pi * 2 * ph) * 0.4
        x = _lp_fast(x, 0.35) * _env(n, 0.02, 2.0)
    elif tonal:
        # long tone: FM bell / siren sweep
        kind = "tone"
        ratio = rng.choice([1.5, 2.0, 3.0, 3.5])
        idx = rng.uniform(1, 4) * np.exp(-3 * t / secs)
        sweep = 1 + rng.uniform(-0.3, 0.5) * t / secs
        ph = np.cumsum(base * sweep / rate)
        x = np.sin(2 * np.pi * ph + idx * np.sin(2 * np.pi * ratio * ph)) * _env(n, 0.03, 1.2)
    elif secs < 0.6:
        # hit: noise burst + low thump
        kind = "hit"
        x = _band(rng.standard_normal(n), rate, rng.uniform(300, 900), rng.uniform(2500, 7000)) * _env(n, 0.005, 4)
        x += 0.8 * np.sin(2 * np.pi * np.cumsum((rng.uniform(50, 90) * (1 - 0.5 * t / secs)) / rate)) * _env(n, 0.002, 3)
    else:
        # whoosh / blast: noise through a moving band, swelling then fading
        kind = "whoosh"
        x = rng.standard_normal(n)
        lo, hi = rng.uniform(200, 600), rng.uniform(1500, 5000)
        half = n // 2
        x = np.concatenate([_band(x[:half], rate, lo, hi * 0.6), _band(x[half:], rate, lo * 1.5, hi)])
        x *= np.sin(np.pi * np.clip(t / secs, 0, 1)) ** 0.7
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.9
    return x, kind
