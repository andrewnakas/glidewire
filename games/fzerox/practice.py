"""Voice practice pack (PERSONAL USE: the reference clips come from the user's own ROM extraction; written outside
the repo, never published).

Per voice (announcer, mrzero): practice_<WHO>_call_and_response.wav = for each line: the reference clip (decoded
with the retail codebooks, at 22.05 kHz), 0.3 s, an 80 ms 880 Hz beep, then a gap of 1.5x + 1.5 s to repeat it
in your own voice. SCRIPT.txt lists the lines in track order; lines.json records each slot (sample, frames) so
takes.py can cut a recording of the whole track back into games/fzerox/voices/<sample>.wav.

    python -m games.fzerox.practice <dirty tree> <out dir>
"""
import json
import os
import sys
import wave

import numpy as np

from cleanroom.audio import vadpcm

from .audio import BANK_C, VER, bank_bases, parse_bank

HERE = os.path.dirname(os.path.abspath(__file__))
HZ = 22050


def beep():
    return (0.2 * np.sin(2 * np.pi * 880 * np.arange(int(0.08 * HZ)) / HZ)).astype(np.float32)


def gap_for(frames):
    return int((frames / HZ * 1.5 + 1.5) * HZ)


def wr(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(HZ)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main(argv):
    dirty, out = argv[1], argv[2]
    L = {k: v for k, v in json.load(open(os.path.join(HERE, "voice_lines.json"))).items() if not k.startswith("_")}
    samples, books = parse_bank(open(os.path.join(dirty, BANK_C)).read())
    bases = bank_bases(dirty)
    table = open(os.path.join(dirty, "bin", VER, "audio_table.bin"), "rb").read()
    os.makedirs(out, exist_ok=True)
    tracks, script, manifest = {}, {}, {}
    for name, v in L.items():
        s = samples[name]
        n = s["size"] // 9 * 16
        _, order, npred, coefs = books[(s["font"], s["book"])]
        st = bases[s["bank"]][0] + s["addr"]
        x = vadpcm.decode(table[st:st + s["size"]], {"order": order, "npred": npred, "book": coefs}, n)
        x = x.astype(np.float32) / 32768
        x = np.interp(np.arange(0, len(x) * HZ / s["rate"]) * s["rate"] / HZ, np.arange(len(x)), x).astype(np.float32)
        who = v["who"]
        tracks.setdefault(who, []).extend([x, np.zeros(int(0.3 * HZ), np.float32), beep(),
                                           np.zeros(gap_for(len(x)), np.float32)])
        k = len(manifest.setdefault(who, [])) + 1
        manifest[who].append({"n": k, "sample": name, "text": v["text"], "frames": len(x)})
        script.setdefault(who, []).append(f"{k:03d}  {len(x) / HZ:4.1f}s  \"{v['text']}\"")
    lines = ["F-Zero X voice practice script. For each voice, play practice_<WHO>_call_and_response.wav",
             "and repeat each line after the beep, in character. Record the whole track in one go (any format),",
             "then:  python -m games.fzerox.takes cut <recording> <WHO> <this folder>", ""]
    for who in sorted(tracks):
        wr(os.path.join(out, f"practice_{who}_call_and_response.wav"), np.concatenate(tracks[who]))
        lines += [f"== {who} ({len(script[who])} lines, practice_{who}_call_and_response.wav)"] + script[who] + [""]
    lines += ["These reference clips come from your own ROM: practice only, do not share or commit them."]
    open(os.path.join(out, "SCRIPT.txt"), "w", encoding="utf8").write("\n".join(lines))
    json.dump(manifest, open(os.path.join(out, "lines.json"), "w"), indent=1)
    print(f"practice pack: {sum(len(v) for v in manifest.values())} lines "
          f"({', '.join(f'{w} {len(v)}' for w, v in sorted(manifest.items()))}) -> {out}")


if __name__ == "__main__":
    main(sys.argv)
