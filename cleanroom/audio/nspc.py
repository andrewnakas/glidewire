"""N-SPC sequence reader + writer (the Super Mario World flavour of Nintendo's SPC700
music driver, as reimplemented in C by snesrev/smw: src/smw_spc_player.c).

Data model in ARAM (all pointers absolute, little endian):
  song table   word per slot (slot id 1 = first word); several slots may share a song
  song         list of words: pattern pointer (high byte != 0) | 0x0000 end |
               0x00nn + target word = jump (nn >= 0x80: forever; else repeat nn more times)
  pattern      8 track pointers, channel 0..7 (0 = channel idle)
  track        bytes: [len 01-7F [param 00-7F]] then one command
                 80-C5 note (key = cmd & 0x7F, plus E4 transposition), C6 tie, C7-CF rest,
                 D0-D9 percussion (6-byte drum table: srcn adsr1 adsr2 gain base key),
                 DA.. effect (argument count in ARGS), 00 end of track / subroutine
               param = gate (bits 4-6, table GATE) | velocity (bits 0-3, table VEL)
  A pattern ends when its first track (lowest channel) reaches its 00 byte, so all tracks
  of a pattern should be equally long.

Reader:  ram = load(upload_blob[, ram]); read_song(ram, ptr) -> Song (notes per channel,
         length + loop point in ticks, tempo, instruments, echo).
         to_notes(song) -> [cleanroom.audio.notes.Note] (percussion on channel ch + 8).
Writer:  Track(...) event builder; encode_track(events) -> bytes;
         Bank(base).add_song(patterns, loop_index) ...; Bank.build() -> (bytes, {slot: ptr}).
Timing:  music tick rate = 500 * tempo / 256 per second (tempo = E2 argument + 1).
"""
import struct
from dataclasses import dataclass, field

from cleanroom.audio.notes import Note

# argument bytes per effect command (DA..F2); E5 is listed with 3 although the C player reads 2
ARGS = {0xda: 1, 0xdb: 1, 0xdc: 2, 0xdd: 3, 0xde: 3, 0xdf: 0, 0xe0: 1, 0xe1: 2, 0xe2: 1, 0xe3: 2,
        0xe4: 1, 0xe5: 3, 0xe6: 0, 0xe7: 1, 0xe8: 2, 0xe9: 3, 0xea: 1, 0xeb: 3, 0xec: 3, 0xed: 0,
        0xee: 1, 0xef: 3, 0xf0: 0, 0xf1: 3, 0xf2: 3}
VEL = [8, 18, 27, 36, 44, 53, 62, 71, 81, 90, 98, 107, 125, 143, 161, 179]
GATE = [51, 102, 128, 153, 179, 204, 230, 255]
TIE, REST = 0xc6, 0xc7
SUB = 0xe9


def tick_rate(tempo):
    return 500.0 * tempo / 256.0


def load(blob, ram=None):
    """Apply an upload stream (size, addr, data ... size 0) to a 64 KiB ARAM image."""
    ram = ram if ram is not None else bytearray(65536)
    p = 0
    while p + 2 <= len(blob):
        n, = struct.unpack_from('<H', blob, p)
        if n == 0:
            break
        a, = struct.unpack_from('<H', blob, p + 2)
        ram[a:a + n] = blob[p + 4:p + 4 + n]
        p += 4 + n
    return ram


def upload(blocks):
    """[(addr, bytes)] -> upload stream."""
    out = bytearray()
    for a, d in blocks:
        out += struct.pack('<HH', len(d), a) + bytes(d)
    return bytes(out + b'\0\0')


def word(ram, a):
    return ram[a] | ram[(a + 1) & 0xffff] << 8


# ------------------------------------------------------------------ reader

@dataclass
class Ev:
    t: int          # tick
    ch: int
    kind: str       # note | perc | fx
    key: int = 0    # note: key incl. transposition; perc: drum index; fx: command
    dur: int = 0
    vel: int = 0    # 0..15 index
    gate: int = 0
    args: tuple = ()


@dataclass
class Song:
    ptr: int
    events: list = field(default_factory=list)
    length: int = 0           # ticks until the end (or until the loop jump)
    loop: int = None          # tick the song jumps back to (None = it ends)
    tempo: int = 0            # first tempo (E2 arg + 1); 0x36 if never set
    patterns: int = 0         # patterns played
    channels: int = 0         # bit mask of channels with notes
    instruments: set = field(default_factory=set)
    echo: tuple = None        # (EON mask, EDL) if echo is used
    errors: list = field(default_factory=list)


def _track(ram, ptr, ch, t0, st, out, limit=4096):
    """Decode one track from ptr; returns its duration. st carries len/vel/gate/transpose."""
    t = 0
    sub = None            # (return ptr, start ptr, loops left)
    p = ptr
    steps = 0
    while steps < 20000:
        steps += 1
        c = ram[p]
        p += 1
        if c == 0:
            if sub is None:
                return t
            ret, start, left = sub
            left -= 1
            if left <= 0:
                p, sub = ret, None
            else:
                p, sub = start, (ret, start, left)
            continue
        if c < 0x80:
            st['len'] = c
            c = ram[p]
            p += 1
            if c < 0x80:
                st['gate'], st['vel'] = c >> 4 & 7, c & 15
                c = ram[p]
                p += 1
        if c >= 0xda:
            n = ARGS.get(c)
            if n is None:
                out.errors.append('bad cmd %02X at %04X' % (c, p - 1))
                return t
            a = tuple(ram[p:p + n])
            p += n
            if c == SUB:
                sub = (p, a[0] | a[1] << 8, a[2])
                p = sub[1]
                continue
            if c == 0xe4:
                st['tr'] = a[0]
            if c == 0xe2 and not out.tempo:
                out.tempo = a[0] + 1
            if c == 0xda:
                out.instruments.add(a[0] + 1)
            if c == 0xef:
                st['eon'] = a[0]
            if c == 0xf1:
                st['edl'] = a[0]
            out.events.append(Ev(t0 + t, ch, 'fx', c, args=a))
            continue
        d = st['len'] or 1
        if c < TIE:
            out.events.append(Ev(t0 + t, ch, 'note', ((c & 0x7f) + st['tr']) & 0xff, d, st['vel'], st['gate']))
        elif c == TIE:
            for e in reversed(out.events):
                if e.ch == ch and e.kind in ('note', 'perc'):
                    if e.t + e.dur == t0 + t:
                        e.dur += d
                    break
        elif c >= 0xd0:
            out.events.append(Ev(t0 + t, ch, 'perc', c - 0xd0, d, st['vel'], st['gate']))
        t += d
        if t > limit * 64:
            break
    out.errors.append('runaway track %04X' % ptr)
    return t


def read_song(ram, ptr):
    s = Song(ptr)
    st = [dict(len=0, vel=15, gate=7, tr=0) for _ in range(8)]
    tr = 0
    p = ptr
    t = 0
    seen = {}             # song list position -> tick first played
    count = 0
    for _ in range(4096):
        w = word(ram, p)
        p += 2
        if w == 0:
            s.length = t
            break
        if w >> 8 == 0:
            tgt = word(ram, p)
            p += 2
            count = (count - 1) & 0xff
            if count == 0:
                continue
            if count & 0x80:
                count = w
            if tgt in seen and (w >= 0x80):
                s.length, s.loop = t, seen[tgt]
                break
            p = tgt
            continue
        seen.setdefault(p - 2, t)
        durs = []
        for ch in range(8):
            tp = word(ram, w + 2 * ch)
            if tp >> 8 == 0:
                continue
            st[ch]['tr'] = tr
            d = _track(ram, tp, ch, t, st[ch], s)
            tr = st[ch]['tr']
            durs.append(d)
        if not durs:
            s.errors.append('empty pattern %04X' % w)
            break
        plen = min(durs)
        s.events = [e for e in s.events if e.t < t + plen]
        for e in s.events:
            if e.t + e.dur > t + plen and e.t >= t:
                e.dur = t + plen - e.t
        t += plen
        s.patterns += 1
    else:
        s.errors.append('song list runaway')
        s.length = t
    for e in s.events:
        if e.kind in ('note', 'perc'):
            s.channels |= 1 << e.ch
    eon = [x.get('eon') for x in st if x.get('eon') is not None]
    edl = [x.get('edl') for x in st if x.get('edl') is not None]
    if eon:
        s.echo = (eon[0], edl[0] if edl else None)
    s.tempo = s.tempo or 0x36
    return s


def to_notes(song, key_offset=24):
    """Notes for MIDI export: key = N-SPC key + key_offset; drums on channel ch + 8 (key 35 + index)."""
    out = []
    for e in song.events:
        v = max(1, min(127, VEL[e.vel] * 127 // 179))
        if e.kind == 'note':
            out.append(Note(e.t, e.ch, max(0, min(127, e.key + key_offset)), v, max(1, e.dur)))
        elif e.kind == 'perc':
            out.append(Note(e.t, e.ch + 8, 35 + e.key, v, max(1, e.dur)))
    return sorted(out)


# ------------------------------------------------------------------ writer

def encode_track(events, end=True):
    """events: list of tuples
         ('n', key, dur[, vel, gate])  note (key 0..0x45 before E4 transposition)
         ('p', drum, dur[, vel, gate]) percussion D0+drum
         ('r', dur)  rest        ('t', dur)  tie
         ('x', cmd, *args)          effect
    vel 0..15 / gate 0..7 default to the previous ones (15/7 at the start).
    The first note always carries an explicit length + param byte, so a track or a
    subroutine body never depends on what played before it."""
    out = bytearray()
    cur = [None, None, None]          # len, vel, gate
    vel, gate = 15, 7
    for e in events:
        k = e[0]
        if k == 'x':
            assert e[1] in ARGS and len(e) - 2 == ARGS[e[1]], e
            out += bytes(e[1:])
            continue
        if k in ('n', 'p'):
            if len(e) > 3:
                vel, gate = e[3], e[4]
            code = (0x80 + e[1]) if k == 'n' else (0xd0 + e[1])
            assert (k == 'n' and 0 <= e[1] <= 0x45) or (k == 'p' and 0 <= e[1] <= 9), e
            dur = e[2]
        else:
            code = REST if k == 'r' else TIE
            dur = e[1]
        first = True
        while dur > 0:
            d = min(dur, 0x60)
            c = code if first or code == REST else TIE
            sound = c not in (TIE, REST)
            if cur[0] != d or (sound and (cur[1], cur[2]) != (vel, gate)):
                out.append(d)
                cur[0] = d
                if sound and (cur[1], cur[2]) != (vel, gate):
                    out.append(gate << 4 | vel)
                    cur[1], cur[2] = vel, gate
            out.append(c)
            first = False
            dur -= d
    if end:
        out.append(0)
    return bytes(out)


def track_ticks(events):
    return sum(e[2] if e[0] in ('n', 'p') else e[1] for e in events if e[0] != 'x')


class Bank:
    """Lays out songs at base: slot table, song lists, patterns, shared tracks/subroutines.
    add_song(patterns, loop=k) with patterns = [{ch: [events]}] (k = index of the pattern
    the song loops back to, None = it ends); alias(slot, other_slot)."""

    def __init__(self, base, nslots):
        self.base, self.nslots = base, nslots
        self.songs = {}           # slot -> (patterns, loop)
        self.alias_of = {}

    def add_song(self, slot, patterns, loop=None):
        self.songs[slot] = (patterns, loop)

    def alias(self, slot, other):
        self.alias_of[slot] = other

    @staticmethod
    def _split_repeat(evs):
        """events = unit * k (k >= 2)? -> (unit, k)."""
        n = len(evs)
        for u in range(1, n // 2 + 1):
            if n % u == 0 and evs[:u] * (n // u) == evs and any(e[0] != 'x' for e in evs[:u]):
                return evs[:u], n // u
        return evs, 1

    def build(self):
        tracks = {}               # bytes -> offset key
        order_tracks = []
        subs = {}

        def track_id(evs):
            unit, k = self._split_repeat(list(evs))
            if k > 1:
                body = encode_track(unit)
                if body not in subs:
                    subs[body] = len(subs)
                    order_tracks.append(('sub', body))
                data = ('call', body, k)
            else:
                data = ('raw', encode_track(evs))
            if data not in tracks:
                tracks[data] = len(tracks)
                order_tracks.append(('trk', data))
            return data

        pats = {}
        song_lists = {}
        for slot in sorted(self.songs):
            patterns, loop = self.songs[slot]
            ids = []
            for pt in patterns:
                key = tuple((ch, track_id(tuple(pt[ch]))) if pt.get(ch) else (ch, None) for ch in range(8))
                if key not in pats:
                    pats[key] = len(pats)
                ids.append(key)
            song_lists[slot] = (ids, loop)
        # layout: table | song lists | patterns | subroutines + tracks
        addr = self.base + 2 * self.nslots
        song_addr = {}
        for slot in sorted(song_lists):
            ids, loop = song_lists[slot]
            song_addr[slot] = addr
            addr += 2 * len(ids) + (6 if loop is not None else 2)
        pat_addr = {}
        for key in pats:
            pat_addr[key] = addr
            addr += 16
        sub_addr, trk_addr = {}, {}
        blobs = []
        for kind, data in order_tracks:
            if kind == 'sub':
                sub_addr[data] = addr
                blobs.append(data)
                addr += len(data)
        for kind, data in order_tracks:
            if kind == 'trk':
                trk_addr[data] = addr
                if data[0] == 'call':
                    a = sub_addr[data[1]]
                    b = bytes([SUB, a & 255, a >> 8, data[2], 0])
                else:
                    b = data[1]
                blobs.append(b)
                addr += len(b)
        out = bytearray()
        ptrs = {}
        for s in range(1, self.nslots + 1):
            src = s
            while src in self.alias_of:
                src = self.alias_of[src]
            ptrs[s] = song_addr.get(src, 0)
            out += struct.pack('<H', ptrs[s])
        for slot in sorted(song_lists):
            ids, loop = song_lists[slot]
            for key in ids:
                out += struct.pack('<H', pat_addr[key])
            if loop is not None:
                out += struct.pack('<HH', 0x00ff, song_addr[slot] + 2 * loop)
            out += b'\0\0'
        for key in pats:
            for ch, tid in key:
                out += struct.pack('<H', trk_addr[tid] if tid is not None else 0)
        for b in blobs:
            out += b
        assert len(out) == addr - self.base
        return bytes(out), ptrs
