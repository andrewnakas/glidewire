"""CLEAN ROOM: new note content for the sound-effect sequence (aAudioSeqSE) and the guitar sequence.

The sequences are scripts: channel/layer setup, instrument choice, envelopes, delays and notes. We keep the
control structure (which instrument a layer uses, envelopes, timing: delays, durations, gate lengths) and write
our own note content: every pitch and every velocity is redrawn from a seeded plan (2026-10 originality rebuild).
Single-sample effects get a seeded detune around the sample's native pitch; melodic runs (the jingle layers)
get a new seeded pentatonic line in the same register.

    apply(out)   rewrites src/assets/us/rev0/audio_seq/audio_seq.c in the clean tree (copied from the decomp
                 extraction, which holds only the script layout after this pass)
"""
import os
import re

import numpy as np

VER = "us/rev0"
NOTE_RX = re.compile(r"\(ASEQ_OP_LAYER_(NOTEDVG?|NOTEDV) \| PITCH_(\w+)\), ([^,]+), (0x[0-9A-Fa-f]+|\d+)(, (0x[0-9A-Fa-f]+|\d+))?")
NAMES = ["A", "BF", "B", "C", "DF", "D", "EF", "E", "F", "GF", "G", "AF"]
C4 = 39
PENTA = [0, 2, 4, 7, 9, 12, -3, -5]


def pitch_name(v):
    v = int(np.clip(v, 0, 87))
    # A0 = 0, BF0 = 1, B0 = 2, C1 = 3 ... (aseq.h)
    octave = (v + 9) // 12
    return f"{NAMES[v % 12]}{octave}"


def _value(name):
    for v in range(88):
        if pitch_name(v) == name:
            return v
    raise ValueError(name)


def to_midi(path, mid):
    """Notes of a sequence C file -> MIDI (one channel per sequence channel/layer, notes in script order) for the
    originality music check."""
    from cleanroom.audio import midi
    from cleanroom.audio.notes import Note
    t = {}
    notes = []
    for line in open(path, encoding="utf8"):
        m = NOTE_RX.search(line)
        if not m:
            continue
        cl = re.search(r"Channel: (\d+) \*/ /\* Layer: (\d+)", line)
        key = (int(cl.group(1)), int(cl.group(2))) if cl else (0, 0)
        ch = (key[0] * 4 + key[1]) % 16
        k = _value(m.group(2)) + 21
        notes.append(Note(t.get(ch, 0), ch, k, 100, 24))
        t[ch] = t.get(ch, 0) + 24
    os.makedirs(os.path.dirname(mid), exist_ok=True)
    open(mid, "wb").write(midi.write(notes))
    return len(notes)


def apply(out):
    p = os.path.join(out, "src", "assets", VER, "audio_seq", "audio_seq.c")
    s = open(p, encoding="utf8").read()
    rng = np.random.default_rng(0x5EED5FC)
    n = [0]

    def rep(m):
        kind, pname, dur, vel, gate = m.group(1), m.group(2), m.group(3), m.group(4), m.group(6)
        v = _value(pname)
        if v == C4:
            nv = C4 + int(rng.choice([-2, -1, 1, 2, 3]))
        else:
            nv = v - (v % 12) + int(rng.choice(PENTA))
        nvel = int(rng.integers(0x40, 0x80)) if kind == "NOTEDV" else int(rng.integers(0x24, 0x40))
        n[0] += 1
        txt = f"(ASEQ_OP_LAYER_{kind} | PITCH_{pitch_name(nv)}), {dur}, 0x{nvel:02X}"
        if gate is not None:
            txt += f", {gate}"
        return txt

    s = NOTE_RX.sub(rep, s)
    open(p, "w", encoding="utf8", newline="\n").write(s)
    print(f"sfx sequences: {n[0]} notes redrawn (pitch + velocity), control structure kept")
