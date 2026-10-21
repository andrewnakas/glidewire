"""Standard MIDI files (format 0/1): read notes, write notes.

    read(data) -> (division, [Note], meta)     meta: tempo (usec/qn), loop (start, end) ticks or None
    write(notes, division, tempo, loop=None, programs=None) -> bytes

Loop points use marker metas "loopStart" / "loopEnd" (RPG-Maker style),
which most N64 MIDI converters understand.
"""
import struct

from cleanroom.audio.notes import Note


def _varlen(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def _read_var(d, p):
    v = 0
    while True:
        b = d[p]
        p += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, p


def read(data: bytes):
    assert data[:4] == b"MThd", "not a MIDI file"
    hlen, fmt, ntrk, division = struct.unpack_from(">IHHH", data, 4)
    pos = 8 + hlen
    notes, meta = [], {"tempo": 500000, "loop": None}
    loop = [None, None]
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            break
        ln = struct.unpack_from(">I", data, pos + 4)[0]
        p, end = pos + 8, pos + 8 + ln
        pos = end
        t, status, on = 0, 0, {}
        while p < end:
            dt, p = _read_var(data, p)
            t += dt
            b = data[p]
            if b == 0xFF:
                typ = data[p + 1]
                n, p = _read_var(data, p + 2)
                body = data[p:p + n]
                p += n
                if typ == 0x51 and n == 3:
                    meta["tempo"] = (body[0] << 16) | (body[1] << 8) | body[2]
                elif typ in (0x06, 0x01):
                    txt = body.decode("latin1").lower()
                    if txt == "loopstart":
                        loop[0] = t
                    elif txt == "loopend":
                        loop[1] = t
                if typ == 0x2F:
                    break
                continue
            if b in (0xF0, 0xF7):
                n, p = _read_var(data, p + 1)
                p += n
                continue
            if b & 0x80:
                status = b
                p += 1
            kind, ch = status & 0xF0, status & 0xF
            if kind in (0xC0, 0xD0):
                p += 1
                continue
            k, v = data[p], data[p + 1]
            p += 2
            if kind == 0x90 and v:
                on.setdefault((ch, k), []).append((t, v))
            elif kind == 0x80 or (kind == 0x90 and not v):
                stack = on.get((ch, k))
                if stack:
                    t0, v0 = stack.pop(0)
                    notes.append(Note(t0, ch, k, v0, max(1, t - t0)))
    if loop[0] is not None:
        meta["loop"] = (loop[0], loop[1])
    notes.sort()
    return division, notes, meta


def _track(events):
    out, now = bytearray(), 0
    for t, _, body in sorted(events, key=lambda e: (e[0], e[1])):
        out += _varlen(t - now) + body
        now = t
    out += b"\x00\xFF\x2F\x00"
    return b"MTrk" + struct.pack(">I", len(out)) + bytes(out)


def write(notes, division=48, tempo=500000, loop=None, programs=None) -> bytes:
    """notes: iterable of Note; one track per channel plus a conductor track.
    loop: (start, end) ticks; programs: {channel: program}."""
    cond = [(0, 0, b"\xFF\x51\x03" + tempo.to_bytes(3, "big"))]
    if loop:
        for t, name in zip(loop, (b"loopStart", b"loopEnd")):
            if t is not None:
                cond.append((t, 1, b"\xFF\x06" + _varlen(len(name)) + name))
    by = {}
    for n in notes:
        by.setdefault(n.ch, []).append(n)
    tracks = [_track(cond)]
    for ch in sorted(by):
        ev = []
        if programs and ch in programs:
            ev.append((0, 0, bytes([0xC0 | ch, programs[ch] & 0x7F])))
        for n in by[ch]:
            ev.append((n.time, 2, bytes([0x90 | ch, n.key & 0x7F, max(1, n.vel & 0x7F)])))
            ev.append((n.time + n.dur, 1, bytes([0x80 | ch, n.key & 0x7F, 0])))
        tracks.append(_track(ev))
    hdr = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division)
    return hdr + b"".join(tracks)
