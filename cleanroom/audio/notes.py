"""Note events shared by the sequence readers/writers and the composer.

    Note(time, ch, key, vel, dur)    ticks, channel, MIDI key, velocity, ticks
    load(path) -> (division, [Note])  by extension: .mid/.midi, .m64, else cseq
"""
from collections import namedtuple

Note = namedtuple("Note", "time ch key vel dur")


def from_cseq(data: bytes):
    from cleanroom.audio import cseq
    division, tracks = cseq.decode(data)
    notes = [Note(e[0], e[2], e[3], e[4], e[5]) for ev in tracks.values() for e in ev if e[1] == "note"]
    return division, sorted(notes)


def from_bytes(data: bytes, fmt: str):
    fmt = fmt.lower().lstrip(".")
    if fmt in ("mid", "midi"):
        from cleanroom.audio import midi
        d, n, _ = midi.read(data)
        return d, n
    if fmt == "m64":
        from cleanroom.audio import m64
        return m64.parse(data)
    return from_cseq(data)


def load(path, fmt=None):
    with open(path, "rb") as f:
        data = f.read()
    if fmt is None:
        fmt = path.rsplit(".", 1)[-1] if "." in path else "cseq"
        if data[:4] == b"MThd":
            fmt = "mid"
    return from_bytes(data, fmt)
