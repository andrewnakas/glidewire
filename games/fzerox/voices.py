"""Placeholder voices: the announcer and Max Volume lines (our own wording) spoken by Piper TTS (our own performances;
no original audio, no cloning), fitted to each sample slot's length and playback rate.

voice_lines.json lists the spoken samples and their words. Each line is spoken a few times with different
speeds; the take Whisper reads back best (and that fits the slot) wins. Writes games/fzerox/voices/<sample>.wav
(mono, 32 kHz); audio.py resamples it to the slot rate and uses it instead of the resynthesised outline.

    python -m games.fzerox.voices build [sample ...]
"""
import json
import os
import re
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "voices")
PIPER = os.environ.get("PIPER_VOICES", "C:/Users/andre/n64work/piper_voices")
WRATE = 32000

# who -> (piper model, semitones, energy boost)
CAST = {"announcer": ("en_US-ryan-high", -1.0, 1.3), "maxvolume": ("en_US-joe-medium", 2.0, 1.1)}
SPEEDS = [0.8, 0.9, 1.0, 1.15]

_V = {}


def piper(model, text, length):
    from piper import PiperVoice, SynthesisConfig
    if model not in _V:
        _V[model] = PiperVoice.load(os.path.join(PIPER, model + ".onnx"))
    v = _V[model]
    cfg = SynthesisConfig(length_scale=length, noise_scale=0.7, noise_w_scale=0.8)
    x = np.concatenate([c.audio_float_array for c in v.synthesize(text, syn_config=cfg)]).astype(np.float64)
    return x, v.config.sample_rate


def trim(x, thr=0.01):
    idx = np.nonzero(np.abs(x) > thr)[0]
    return x[max(0, idx[0] - 200):idx[-1] + 200] if len(idx) else x[:0]


def resample(x, sr, hz, n=None):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr), int(hz))
    y = resample_poly(x, int(hz) // g, int(sr) // g)
    if n is not None:
        y = np.concatenate([y, np.zeros(max(0, n - len(y)))])[:n]
    return y


def pitch(x, sr, semis):
    """Pitch shift by resampling then time-restoring (formants move too: a different voice colour)."""
    if not semis:
        return x
    f = 2 ** (semis / 12.0)
    y = np.interp(np.arange(0, len(x) - 1, f), np.arange(len(x)), x)
    return y


_W = None


def hear(x, sr):
    global _W
    from faster_whisper import WhisperModel
    if _W is None:
        _W = WhisperModel("base.en", device="cpu", compute_type="int8")
    y = resample(x, sr, 16000).astype(np.float32)
    segs, _ = _W.transcribe(y, language="en", beam_size=3)
    return " ".join(s.text for s in segs)


def norm(t):
    return re.sub(r"[^a-z ]", "", t.lower()).split()


def score(heard, text):
    a, b = norm(heard), norm(text)
    if not b:
        return 0.0
    return sum(1 for w in b if w in a) / len(b)


def write_wav(p, x, sr):
    y = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(p, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(y.tobytes())


def read_wav(p, rate=None):
    with wave.open(p) as f:
        sr = f.getframerate()
        x = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float64) / 32768.0
    return resample(x, sr, rate) if rate and rate != sr else x


def build(only=None):
    lines = {k: v for k, v in json.load(open(os.path.join(HERE, "voice_lines.json"))).items() if not k.startswith("_")}
    spec = json.load(open(os.path.join(HERE, "spec", "audio.json")))["samples"]
    os.makedirs(OUT, exist_ok=True)
    for name, ln in lines.items():
        if only and name not in only:
            continue
        d = spec[name]
        slot_secs = d["n"] / d["rate"]
        model, semis, boost = CAST[ln["who"]]
        best = None
        for sp in SPEEDS:
            x, sr = piper(model, ln["text"], sp)
            x = trim(pitch(x, sr, semis))
            secs = len(x) / sr
            if secs > slot_secs * 0.98:
                # too long: speed up by resampling time (and pitch) just enough
                x = np.interp(np.linspace(0, len(x) - 1, int(slot_secs * 0.97 * sr)), np.arange(len(x)), x)
            sc = score(hear(x, sr), ln["text"])
            fit = min(1.0, len(x) / sr / slot_secs)
            key = (round(sc, 2), fit)
            if best is None or key > best[0]:
                best = (key, x, sr)
        _, x, sr = best
        x = np.tanh(x / (np.abs(x).max() or 1) * boost) * 0.9
        y = resample(x, sr, WRATE)
        write_wav(os.path.join(OUT, name + ".wav"), y, WRATE)
        print(f"{name:28s} {ln['who']:9s} heard {best[0][0]:.2f} fit {best[0][1]:.2f}  {ln['text']}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2:] or None)
