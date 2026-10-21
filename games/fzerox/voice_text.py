"""DIRTY ROOM: which samples are spoken lines, and what they say (words only -> spec/voice_text.json).

    python -m games.fzerox.voice_text <dirty tree>

Decodes each short sample (0.3-5 s) in memory, runs faster-whisper (small.en) and keeps the transcript and
the no-speech probability. No audio is written anywhere. The words are facts (like dialogue text); the
placeholder voices (voices.py) and the practice pack speak them.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import resample_poly

from cleanroom.audio import vadpcm

from .audio import BANK_C, SPEC, VER, bank_bases, parse_bank


def main(dirty):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    samples, books = parse_bank(open(os.path.join(dirty, BANK_C)).read())
    bases = bank_bases(dirty)
    table = open(os.path.join(dirty, "bin", VER, "audio_table.bin"), "rb").read()
    out = {}
    for name, s in sorted(samples.items()):
        n = s["size"] // 9 * 16
        secs = n / s["rate"]
        if not 0.3 <= secs <= 5.0:
            continue
        _, order, npred, coefs = books[(s["font"], s["book"])]
        st = bases[s["bank"]][0] + s["addr"]
        pcm = vadpcm.decode(table[st:st + s["size"]], {"order": order, "npred": npred, "book": coefs}, n)
        x = pcm.astype(np.float32) / 32768.0
        x16 = resample_poly(x, 16000, s["rate"]).astype(np.float32)
        segs, info = model.transcribe(x16, language="en", beam_size=5, vad_filter=False)
        segs = list(segs)
        text = " ".join(sg.text.strip() for sg in segs).strip()
        nsp = min((sg.no_speech_prob for sg in segs), default=1.0)
        lp = max((sg.avg_logprob for sg in segs), default=-9.0)
        out[name] = {"text": text, "no_speech": round(float(nsp), 3), "logprob": round(float(lp), 2),
                     "secs": round(secs, 2), "rate": s["rate"]}
        print(f"{name:28s} {secs:5.2f}s ns={nsp:.2f} lp={lp:5.2f}  {text}")
    json.dump(out, open(os.path.join(SPEC, "voice_text.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
