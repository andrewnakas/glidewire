"""SM64-family sequence-player scripts (.m64): static note reader + writer.

    parse(data) -> (48, [Note])      48 tatums per beat; key = pitch + 21
    write(notes, bpm, length, loop=None, programs=None) -> bytes

The reader walks seq -> channel -> layer scripts statically (calls are
followed, backward jumps end a stream, loops play once, branches fall
through) and accumulates delays into onset times. It is meant for
similarity checks, not playback: unknown opcodes end that stream.
The writer emits large-notes mode, one layer per voice (max 4 per
channel), a seq script that starts every channel and idles, and
per-layer loop jumps at the loop tick.
"""
import struct

from cleanroom.audio.notes import Note

TATUMS = 48
KEY0 = 21          # pitch 0 = A0
VOICES = 4

# argument byte counts; "cu" = compressed u16
SEQ = {0xFE: 0, 0xFD: "cu", 0xF8: 1, 0xF7: 0, 0xF2: 1, 0xF1: 0, 0xDF: 1, 0xDE: 1, 0xDD: 1, 0xDC: 1,
       0xDB: 1, 0xDA: 1, 0xD7: 2, 0xD6: 2, 0xD5: 1, 0xD4: 0, 0xD3: 1, 0xD2: 2, 0xD1: 2, 0xD0: 1,
       0xCC: 1, 0xC9: 1, 0xC8: 1}
CHAN = {0xFE: 0, 0xFD: "cu", 0xF8: 1, 0xF7: 0, 0xF6: 0, 0xF2: 1, 0xF1: 0, 0xEB: 2, 0xE9: 1, 0xE8: 8,
        0xE7: 2, 0xE6: 1, 0xE5: 1, 0xE4: 0, 0xE3: 1, 0xE2: 3, 0xE1: 3, 0xE0: 1, 0xDF: 1, 0xDE: 2,
        0xDD: 1, 0xDC: 1, 0xDB: 1, 0xDA: 2, 0xD9: 1, 0xD8: 1, 0xD7: 1, 0xD6: 1, 0xD4: 1, 0xD3: 1,
        0xD2: 1, 0xD1: 1, 0xD0: 1, 0xCC: 1, 0xC9: 1, 0xC8: 1, 0xC7: 3, 0xC6: 1, 0xC5: 0, 0xC4: 0,
        0xC3: 0, 0xC2: 2, 0xC1: 1}
CHAN_LOW = {0x00: 0, 0x10: 2, 0x20: 0, 0x30: 1, 0x40: 1, 0x50: 0, 0x60: 0, 0x70: 0, 0x80: 0,
            0xA0: 0, 0xB0: 0}
LAYER = {0xF8: 1, 0xF7: 0, 0xF4: 1, 0xC1: 1, 0xC3: "cu", 0xC4: 0, 0xC5: 0, 0xC6: 1, 0xC8: 0,
         0xC9: 1, 0xCA: 1, 0xCB: 3, 0xCC: 0}
JUMPS = (0xFC, 0xFB, 0xFA, 0xF9, 0xF5)


def _cu(d, p):
    b = d[p]
    if b & 0x80:
        return ((b & 0x7F) << 8) | d[p + 1], p + 2
    return b, p + 1


def _skip(d, p, n):
    if n == "cu":
        return _cu(d, p)
    return (int.from_bytes(d[p:p + n], "big") if n else 0), p + n


def _flow(d, p, cmd, stack, seen):
    """Handle jumps/calls/end. Returns the new pc or None to stop."""
    if cmd == 0xFF:
        return stack.pop() if stack else None
    tgt = struct.unpack_from(">H", d, p)[0]
    if cmd == 0xFC:
        stack.append(p + 2)
        return tgt if len(stack) < 8 else None
    if cmd == 0xFB:
        return tgt if tgt > p and tgt not in seen else None
    return p + 2                        # conditional branches fall through


def _layer(d, p, t0, ch, transpose, notes, limit=20000):
    stack, seen, t = [], set(), t0
    large, vel, pp, short_pp, ltr = True, 80, 0, 0x80, 0
    for _ in range(limit):
        if p is None or p >= len(d):
            return
        seen.add(p)
        cmd = d[p]
        p += 1
        if cmd in (0xFF, 0xFC, 0xFB):
            p = _flow(d, p, cmd, stack, seen)
        elif cmd == 0xC0:
            v, p = _cu(d, p)
            t += v
        elif cmd == 0xC2:
            ltr = struct.unpack_from(">b", d, p)[0]
            p += 1
        elif cmd == 0xC7:
            mode = d[p]
            p += 2
            p = p + 1 if mode & 0x80 else _cu(d, p)[1]
        elif cmd in LAYER:
            if cmd == 0xC1:
                vel = d[p]
            if cmd == 0xC3:
                short_pp = _cu(d, p)[0]
            p = _skip(d, p, LAYER[cmd])[1]
        elif cmd >= 0xD0:
            continue
        elif cmd < 0xC0:
            gate, kind = 0, cmd & 0xC0
            if large:
                if kind == 0x00:
                    pp, p = _cu(d, p)
                    vel, gate = d[p], d[p + 1]
                    p += 2
                elif kind == 0x40:
                    pp, p = _cu(d, p)
                    vel = d[p]
                    p += 1
                else:
                    vel, gate = d[p], d[p + 1]
                    p += 2
            elif kind == 0x00:
                pp, p = _cu(d, p)
            elif kind == 0x40:
                pp = short_pp
            dur = max(1, pp * gate // 256) if gate else max(1, pp)
            notes.append(Note(t, ch, (cmd & 0x3F) + KEY0 + ltr + transpose, vel, dur))
            t += pp
        else:
            return


def _channel(d, p, t0, ch, transpose, notes, limit=20000):
    stack, seen, t = [], set(), t0
    for _ in range(limit):
        if p is None or p >= len(d):
            return
        seen.add(p)
        cmd = d[p]
        p += 1
        if cmd in (0xFF,) + JUMPS:
            p = _flow(d, p, cmd, stack, seen)
        elif cmd == 0xFD:
            v, p = _cu(d, p)
            t += v
        elif cmd == 0xFE:
            t += 1
        elif cmd == 0xF3:
            return
        elif cmd == 0xDB:
            transpose = struct.unpack_from(">b", d, p)[0]
            p += 1
        elif cmd in CHAN:
            p = _skip(d, p, CHAN[cmd])[1]
        elif cmd & 0xF0 == 0x90:
            off = struct.unpack_from(">H", d, p)[0]
            p += 2
            _layer(d, off, t, ch, transpose, notes)
        elif cmd & 0xF0 in CHAN_LOW:
            p = _skip(d, p, CHAN_LOW[cmd & 0xF0])[1]
        else:
            return


def parse(data: bytes, limit=20000):
    d, notes = bytes(data), []
    stack, seen, p, t, transpose = [], set(), 0, 0, 0
    started = set()
    for _ in range(limit):
        if p is None or p >= len(d):
            break
        seen.add(p)
        cmd = d[p]
        p += 1
        if cmd in (0xFF,) + JUMPS:
            p = _flow(d, p, cmd, stack, seen)
        elif cmd == 0xFD:
            v, p = _cu(d, p)
            t += v
        elif cmd == 0xFE:
            t += 1
        elif cmd == 0xDF:
            transpose = struct.unpack_from(">b", d, p)[0]
            p += 1
        elif cmd in SEQ:
            p = _skip(d, p, SEQ[cmd])[1]
        elif cmd & 0xF0 == 0x90:
            off = struct.unpack_from(">H", d, p)[0]
            p += 2
            if (cmd & 0xF, off) not in started:
                started.add((cmd & 0xF, off))
                _channel(d, off, t, cmd & 0xF, transpose, notes)
        elif cmd & 0xF0 in (0x00, 0x50, 0x70, 0x80):
            continue
        else:
            break
    return TATUMS, sorted(notes)


# ------------------------------------------------------------------ writer

def _cu_bytes(v):
    v = max(0, min(0x7FFF, int(v)))
    return bytes([v]) if v < 0x80 else bytes([0x80 | (v >> 8), v & 0xFF])


def _voices(notes):
    """Split one channel's notes into monophonic voices (greedy)."""
    ends, out = [], []
    for n in sorted(notes, key=lambda n: (n.time, -n.key)):
        for i, e in enumerate(ends):
            if e <= n.time:
                ends[i] = n.time + n.dur
                out[i].append(n)
                break
        else:
            if len(out) < VOICES:
                ends.append(n.time + n.dur)
                out.append([n])
    return out


def _rest(t, op=0xC0):
    out = b""
    while t > 0:
        step = min(t, 0x7FFF)
        out += bytes([op]) + _cu_bytes(step)
        t -= step
    return out


def _layer_bytes(voice, length, loop):
    """One voice -> (layer bytes without terminator, loop label offset)."""
    keys = [n.key for n in voice] or [KEY0]
    tr = max(-128, min(127, min(keys) - KEY0)) if max(keys) - KEY0 > 63 or min(keys) < KEY0 else 0
    out = bytearray(b"\xC2" + struct.pack(">b", tr)) if tr else bytearray()
    label, t = None, 0
    for i, n in enumerate(voice):
        if loop is not None and label is None and n.time >= loop:
            out += _rest(loop - t)
            t, label = loop, len(out)
        out += _rest(n.time - t)
        t = max(t, n.time)
        nxt = voice[i + 1].time if i + 1 < len(voice) else length
        if loop is not None and label is None:
            nxt = min(nxt, loop)
        delay = max(1, min(0x7FFF, nxt - t, n.dur))     # exact: gaps become rests
        pitch = max(0, min(63, n.key - KEY0 - tr))
        out += bytes([pitch]) + _cu_bytes(delay) + bytes([max(1, min(127, n.vel)), 0])
        t += delay
    if loop is not None and label is None:
        out += _rest(loop - t)
        t, label = max(t, loop), len(out)
    out += _rest(length - t)
    return bytes(out), label


def write(notes, bpm=120, length=None, loop=None, programs=None, volume=100) -> bytes:
    """notes in 48-tatum ticks; loop = tick the song loops back to (None =
    play once and end). programs: {channel: instrument index}."""
    by = {}
    for n in notes:
        by.setdefault(n.ch, []).append(n)
    length = length or max((n.time + n.dur for n in notes), default=TATUMS)
    chans = sorted(by)[:16]
    term = 3 if loop is not None else 1

    def tail(at):          # idle forever (loop) or wait out the song and end
        if loop is not None:
            return b"\xFD" + _cu_bytes(0x7FFF) + b"\xFB" + struct.pack(">H", at)
        return _rest(length, 0xFD) + b"\xFF"

    head = b"\xD3\x80\xD7" + struct.pack(">H", sum(1 << c for c in chans)) + \
        bytes([0xDB, volume, 0xDD, max(1, min(255, int(bpm)))])
    pos = len(head) + 3 * len(chans) + len(tail(0))
    layouts = []
    for c in chans:
        lays = [_layer_bytes(v, length, loop) for v in _voices(by[c])]
        chead = bytes([0xC4, 0xC1, (programs or {}).get(c, 0) & 0xFF, 0xDF, 127, 0xDD, 64])
        layouts.append((c, pos, chead, lays))
        pos += len(chead) + 3 * len(lays) + len(tail(0)) + sum(len(b) + term for b, _ in lays)
    out = bytearray(head)
    for c, cpos, _, _ in layouts:
        out += bytes([0x90 | c]) + struct.pack(">H", cpos)
    out += tail(len(head) + 3 * len(chans))
    for c, cpos, chead, lays in layouts:
        assert len(out) == cpos
        lpos = cpos + len(chead) + 3 * len(lays) + len(tail(0))
        ch, starts = bytearray(chead), []
        for i, (b, _) in enumerate(lays):
            ch += bytes([0x90 | i]) + struct.pack(">H", lpos)
            starts.append(lpos)
            lpos += len(b) + term
        ch += tail(cpos + len(ch))
        out += ch
        for (b, label), start in zip(lays, starts):
            out += b + (b"\xFB" + struct.pack(">H", start + label) if loop is not None else b"\xFF")
    assert len(out) < 0x10000, "m64 larger than 64 KB"
    return bytes(out)
