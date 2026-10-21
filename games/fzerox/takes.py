"""Cut your recording of a practice track into per-line takes and install them as the game's voices.

    python -m games.fzerox.takes cut <recording.wav|mp3|m4a> <WHO> <practice pack dir>

The recording is of practice_<WHO>_call_and_response.wav played back while you repeat each line after the beep.
We find each beep (880 Hz) in your recording by the track layout from lines.json, take the speech in the gap
after it, trim silence and write games/fzerox/voices/<sample>.wav. Then rebuild the audio:
    python -m games.fzerox.audio voices D:/n64work/fzerox/clean && games/fzerox/build_clean.sh
"""
import json
import os
import sys

import numpy as np

from .practice import HZ, gap_for
from .voices import OUT as _OUT, WRATE, resample, trim, write_wav

OUT = os.environ.get("TAKES_OUT", _OUT)


def load(path):
    import av
    c = av.open(path)
    st = c.streams.audio[0]
    rs = av.AudioResampler(format="flt", layout="mono", rate=HZ)
    chunks = []
    for fr in c.decode(st):
        for f2 in rs.resample(fr):
            chunks.append(f2.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float64)


def beeps(x):
    """Times (samples) of 880 Hz beeps: 10 ms frames whose energy is mostly at 880 Hz (speech never is)."""
    hop = HZ // 100
    n = len(x) // hop
    t = np.arange(hop) / HZ
    c, s = np.cos(2 * np.pi * 880 * t), np.sin(2 * np.pi * 880 * t)
    fr = x[:n * hop].reshape(n, hop)
    tone = (fr @ c) ** 2 + (fr @ s) ** 2
    total = (fr ** 2).sum(1) * hop / 2 + 1e-12
    hit = (tone / total > 0.6) & (np.sqrt(total / hop) > 0.01)
    out, last = [], -10
    for i in np.nonzero(hit)[0]:
        if i - last > 20:
            out.append(i * hop)
        last = i
    return np.asarray(out)


def cut(rec, who, pack):
    lines = json.load(open(os.path.join(pack, "lines.json")))[who]
    x = load(rec)
    # expected beep times from the track layout
    t, offs = 0, []
    for ln in lines:
        t += ln["frames"] + int(0.3 * HZ)
        offs.append(t)
        t += int(0.08 * HZ) + gap_for(ln["frames"])
    found = beeps(x)
    if len(found) >= max(1, len(lines) // 2):
        lead = int(found[0]) - offs[0]
        how = f"{len(found)} beeps heard"
    else:
        # no beeps in the recording (headphones): the first loud sound is the first reply
        env = np.abs(x) > 0.05
        first = int(np.argmax(env)) if env.any() else 0
        lead = first - (offs[0] + int(0.1 * HZ))
        how = "no beeps heard, aligned on the first reply"
    print(f"{who}: {len(lines)} lines, {how}")
    os.makedirs(OUT, exist_ok=True)
    for ln, o in zip(lines, offs):
        b = o + lead
        if len(found):
            near = found[np.abs(found - b) < HZ]
            if len(near):
                b = int(near[0])
        seg = x[max(0, b + int(0.1 * HZ)): b + int(0.1 * HZ) + gap_for(ln["frames"])]
        seg = trim(seg, 0.02)
        if len(seg) < HZ // 10:
            print(f"  {ln['n']:03d} {ln['sample']}: no speech found, skipped")
            continue
        seg = seg / (np.abs(seg).max() or 1) * 0.9
        write_wav(os.path.join(OUT, ln["sample"] + ".wav"), resample(seg, HZ, WRATE), WRATE)
        print(f"  {ln['n']:03d} {ln['sample']}: {len(seg) / HZ:.2f}s  \"{ln['text']}\"")


if __name__ == "__main__":
    if sys.argv[1] == "cut":
        cut(sys.argv[2], sys.argv[3], sys.argv[4])
