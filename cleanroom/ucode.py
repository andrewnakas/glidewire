"""RSP microcode placeholders: tagged stubs instead of retail microcode.

Clean-room ROMs must not carry retail RSP microcode (F3DEX/F3DEX2/S2DEX/L3DEX graphics, aspMain /
n_aspMain audio, the libmus audio ucode, rspboot). HLE emulators never run the microcode anyway: they look
at it only to decide which microcode family to emulate. So each retail segment is replaced by a small
ORIGINAL stub carrying a "CRUCODE1" tag, and the emulator cores we ship are patched to read the tag
first (ports/emu/ucode-tags/). Stub layout (big-endian, same for text and data segments):

    0x00  u32 0x0000000D, u32 0x00000000   RSP `break` + nop: an LLE RSP that runs the stub halts at once
    0x08  "CRUCODE1"                       magic
    0x10  family[16]                       ASCII, NUL padded (FAMILIES below; e.g. "f3dex2", "aspmain")
    0x20  seg[4]                           "TEXT", "DATA" or "BOOT"
    0x24  u16 version                      major*100 + minor (F3DEX2 2.08 -> 208; 0 = unknown)
    0x26  u16 flags                        FLAGS below (NoN, Rej, fifo, xbus, lp)
    0x28  u32 size                         padded size of this segment
    0x2C  u32 crc32 of bytes 0x08..0x2B    header check
    0x30  ASCII note, then zero padding up to `size` (a structural fact: the retail segment size)

Commands (one-screen summaries):
    python -m cleanroom.ucode stub <family> <text|data|boot> <size> -o out.bin [--version 2.08] [--flags fifo]
    python -m cleanroom.ucode stubs <ucode.json> <out dir>      # game spec -> rsp/<name>.textbin.bin etc.
    python -m cleanroom.ucode scan <rom|elf> [--elf E | --map M] # retail ucode left? tags present?
    python -m cleanroom.ucode learn <rom|elf|bin dir> [--elf E | --map M] [--family sym=fam]  # DIRTY side
    python -m cleanroom.ucode swap <rom> -o <out> [--elf E | --map M] [--spec ucode.json] [--crc auto|none]

`learn` reads retail blobs (dirty room only) and records SHORT identifying hashes (16 hex of the sha1
of the first 64 bytes, 12 hex of the whole blob) in ucode_known.json; no retail bytes are stored.
`swap` replaces segments in an uncompressed ROM in place (same size, layout unchanged); games that build
from a decomp should instead write the stubs as the decomp's ucode inputs (`stubs`) and rebuild.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import zlib

MAGIC = b"CRUCODE1"
PRE = struct.pack(">II", 0x0000000D, 0)          # break; nop
HDR = struct.Struct(">8s8s16s4sHHII")            # pre, magic, family, seg, version, flags, size, crc
NOTE = b"clean-room placeholder RSP microcode tag (HLE only); contains no Nintendo code"
KNOWN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ucode_known.json")

FLAGS = {"non": 1, "rej": 2, "fifo": 4, "xbus": 8, "lp": 16}

# family -> (kind, what the patched emulators emulate)
FAMILIES = {
    "boot": ("boot", "rspboot (ignored by HLE)"),
    "f3d": ("gfx", "Fast3D (RSP SW 2.0)"),
    "f3dex": ("gfx", "F3DEX 1.x (flags: non, rej, lp = F3DLX/F3DLP)"),
    "f3dex2": ("gfx", "F3DEX2 2.x (flag non)"),
    "f3dex2_rej": ("gfx", "F3DEX2.Rej / F3DLX2.Rej"),
    "f3dzex2_oot": ("gfx", "F3DZEX2 (OoT era)"),
    "f3dzex2_mm": ("gfx", "F3DZEX2 (MM era)"),
    "l3d": ("gfx", "Line3D"),
    "l3dex": ("gfx", "L3DEX 1.x"),
    "l3dex2": ("gfx", "L3DEX2"),
    "s2dex": ("gfx", "S2DEX 1.x (version 103/105/107)"),
    "s2dex2": ("gfx", "S2DEX2"),
    "turbo3d": ("gfx", "Turbo3D"),
    "zsortp": ("gfx", "ZSort"),
    "f3dex3": ("gfx", "F3DEX3 (open microcode; tag only for uniformity)"),
    "aspmain": ("audio", "audio ABI1 (aspMain; also the libmus games' audio ucode)"),
    "aspmain_ge": ("audio", "audio ABI1, GoldenEye variant"),
    "aspmain_bc": ("audio", "audio ABI1, Blast Corps / DKR variant"),
    "naudio": ("audio", "n_aspMain (nusys n_audio)"),
    "naudio_bk": ("audio", "n_aspMain, Banjo-Kazooie variant"),
    "naudio_dk": ("audio", "n_aspMain, DK64 variant"),
    "naudio_mp3": ("audio", "n_aspMain + MP3 (BT/JFG/PD)"),
    "naudio_cbfd": ("audio", "n_aspMain, Conker variant"),
    "nead_mk": ("audio", "ABI2 MK64 / WaveRace (E)"),
    "nead_sfj": ("audio", "ABI2 StarFox (J)"),
    "nead_sf": ("audio", "ABI2 StarFox / Lylat Wars"),
    "nead_wrjb": ("audio", "ABI2 WaveRace (J RevB)"),
    "nead_fz": ("audio", "ABI2 F-Zero X"),
    "nead_efz": ("audio", "ABI2 F-Zero X Expansion"),
    "nead_ys": ("audio", "ABI2 Yoshi's Story"),
    "nead_1080": ("audio", "ABI2 1080"),
    "nead_oot": ("audio", "ABI2 OoT"),
    "nead_mm": ("audio", "ABI2 MM / Pokemon Stadium 2"),
    "nead_mmb": ("audio", "ABI2 MM (E beta)"),
    "nead_ac": ("audio", "ABI2 Animal Crossing"),
    "nead_mats": ("audio", "ABI2 Mario Artist Talent Studio"),
    "musyx1": ("audio", "MusyX v1"),
    "musyx2": ("audio", "MusyX v2"),
}
ALIASES = {"libmus": "aspmain", "n_aspmain": "naudio", "rspboot": "boot", "f3dex2_fifo": "f3dex2"}

# Dirty-side audio classifier: short hashes of the ABI probe keys mupen64plus-rsp-hle uses
# ("abi1:<word 0x28>", "abi2:<word 0x10>", "abi3:<word 0x10>"); the words themselves are not stored.
AUDIO_SIG = {
    "4fbd6bb060": "aspmain", "b849e710d9": "aspmain_ge", "86c9f844cc": "aspmain_bc",
    "08b41cdc63": "nead_mk", "8732f039b2": "nead_sfj", "570db1f6d5": "nead_wrjb", "6eb40a0482": "nead_sf",
    "ffaf382082": "nead_fz", "26d4a39d1e": "nead_ys", "1a5f598cc8": "nead_1080", "4141f19b3a": "nead_oot",
    "bfffd459f8": "nead_mm", "5d3adb0c22": "nead_mmb", "41966cca08": "nead_ac", "67eeae3cfd": "musyx2",
    "8555e5f126": "nead_mats", "c148ee4c51": "nead_efz", "c90a887fa6": "musyx1", "5eb27a05b7": "naudio",
    "07180a92dc": "naudio_bk", "1030356377": "naudio_dk", "1224960fda": "naudio_mp3", "ed6f6c378c": "naudio_cbfd",
}


# ------------------------------------------------------------------ tag format
def family_name(f):
    f = f.lower()
    f = ALIASES.get(f, f)
    if f not in FAMILIES:
        raise ValueError("unknown ucode family %r (known: %s)" % (f, ", ".join(FAMILIES)))
    return f


def parse_version(v):
    if v in (None, "", 0):
        return 0
    if isinstance(v, int):
        return v
    m = re.match(r"^(\d+)\.(\d+)$", str(v))
    return int(m.group(1)) * 100 + int(m.group(2)) if m else int(v)


def parse_flags(fl):
    if isinstance(fl, int):
        return fl
    if isinstance(fl, str):
        fl = [x for x in re.split(r"[,+ ]", fl) if x]
    out = 0
    for x in fl or ():
        out |= FLAGS[x.lower()]
    return out


def make_stub(family, seg, size, version=0, flags=0):
    """An original placeholder segment of exactly `size` bytes (>= 0x30, multiple of 8 recommended)."""
    family = family_name(family)
    seg = seg.upper()[:4].encode()
    if size < 0x30:
        raise ValueError("stub size %d < 48" % size)
    head = HDR.pack(PRE, MAGIC, family.encode(), seg, parse_version(version), parse_flags(flags), size, 0)
    crc = zlib.crc32(head[8:0x2C]) & 0xFFFFFFFF
    head = head[:0x2C] + struct.pack(">I", crc)
    body = head + NOTE[: max(0, size - len(head))]
    return body + b"\0" * (size - len(body))


def parse_tag(buf, off=0):
    """Tag dict at buf[off:], or None."""
    if len(buf) < off + HDR.size or buf[off + 8: off + 16] != MAGIC:
        return None
    pre, magic, fam, seg, ver, flags, size, crc = HDR.unpack_from(buf, off)
    ok = (zlib.crc32(buf[off + 8: off + 0x2C]) & 0xFFFFFFFF) == crc
    return {"family": fam.rstrip(b"\0").decode("ascii", "replace"), "seg": seg.decode("ascii", "replace"),
            "version": ver, "flags": [k for k, v in FLAGS.items() if flags & v], "size": size, "crc_ok": ok}


def find_tags(rom):
    out = []
    i = rom.find(MAGIC)
    while i >= 0:
        t = parse_tag(rom, i - 8) if i >= 8 else None
        if t:
            t["offset"] = i - 8
            out.append(t)
        i = rom.find(MAGIC, i + 1)
    return out


# ------------------------------------------------------------------ ROM helpers
def to_z64(rom):
    head = rom[:4]
    if head == b"\x37\x80\x40\x12":                       # .v64: 16-bit swapped
        return _swap(rom, "v64"), "v64"
    if head == b"\x40\x12\x37\x80":                       # .n64: 32-bit little endian
        return _swap(rom, "n64"), "n64"
    return rom, "z64"


def from_z64(rom, order):
    """Back to the original byte order (both swaps are their own inverse)."""
    return rom if order == "z64" else _swap(rom, order)


def _swap(rom, order):
    b = bytearray(rom)
    if order == "v64":
        b[0::2], b[1::2] = rom[1::2], rom[0::2]
    else:
        for k in range(4):
            b[k::4] = rom[3 - k::4]
    return bytes(b)


CIC_SEEDS = {"6101/6102": 0xF8CA4DDC, "6103": 0xA3886759, "6105": 0xDF26F436, "6106": 0x1FEA617A}


def n64_crc(rom, cic):
    """libultra boot checksum (CRC1, CRC2) for one CIC family."""
    m = 0xFFFFFFFF
    seed = CIC_SEEDS[cic]
    t1 = t2 = t3 = t4 = t5 = t6 = seed
    data = rom[0x1000:0x101000].ljust(0x100000, b"\0")
    words = struct.unpack(">262144I", data)
    ipl = struct.unpack(">%dI" % 0x100, rom[0x750:0x750 + 0x400].ljust(0x400, b"\0")) if cic == "6105" else None
    for i, d in enumerate(words):
        if (t6 + d) & m < t6:
            t4 = (t4 + 1) & m
        t6 = (t6 + d) & m
        t3 ^= d
        r = ((d << (d & 0x1F)) | (d >> ((32 - (d & 0x1F)) & 0x1F))) & m if d & 0x1F else d
        t5 = (t5 + r) & m
        t2 = (t2 ^ r) if t2 > d else (t2 ^ (t6 ^ d))
        if cic == "6105":
            t1 = (t1 + (ipl[(i * 4 & 0xFF) >> 2] ^ d)) & m
        else:
            t1 = (t1 + (t5 ^ d)) & m
    if cic == "6103":
        return (t6 ^ t4) + t3 & m, (t5 ^ t2) + t1 & m
    if cic == "6106":
        return ((t6 * t4) & m) + t3 & m, ((t5 * t2) & m) + t1 & m
    return t6 ^ t4 ^ t3, t5 ^ t2 ^ t1


def fix_crc(old, new):
    """Recompute the header checksum of `new` with the CIC whose seed reproduces `old`'s checksum."""
    want = struct.unpack_from(">II", old, 0x10)
    for cic in CIC_SEEDS:
        if n64_crc(old, cic) == want:
            c = n64_crc(new, cic)
            return new[:0x10] + struct.pack(">II", *c) + new[0x18:], cic
    return new, None


# ------------------------------------------------------------------ segments from symbols
SYM = re.compile(r"^(\w+?)(Text|Data)(Start|End)$")


def name_family(base):
    """libultra symbol base (gspF3DEX2_fifo, aspMain, n_aspMain, rspboot ...) -> (family, flags) or None."""
    b = base.lower()
    if b.endswith("_boot") or b == "boot":
        return "boot", []
    if not b.startswith(("gsp", "asp", "n_asp", "rspboot")) and re.match(r"^(f3d|l3d|s2d|line3d|fast3d|turbo3d|zsort)", b):
        b = "gsp" + b                               # bare names (F3DEX, F3DLX, S2DEX2 ...) as in mk64
    flags = [f for f in ("fifo", "xbus") if f in b]
    if "non" in re.split(r"[_.]", b)[1:] or b.endswith("non"):
        flags.append("non")
    if b.startswith("rspboot"):
        return "boot", []
    if b.startswith("n_aspmain"):
        return "naudio", []
    if b.startswith("aspmain"):
        return "aspmain", []
    for pre, fam in (("gspf3dzex2", None), ("gspf3dex2", "f3dex2"), ("gspf3dlx2", "f3dex2_rej"),
                     ("gspf3dex", "f3dex"), ("gspf3dlx", "f3dex"), ("gspf3dlp", "f3dex"), ("gspfast3d", "f3d"),
                     ("gspf3dnon", "f3d"), ("gspl3dex2", "l3dex2"), ("gspl3dex", "l3dex"), ("gspline3d", "l3d"),
                     ("gsps2dex2", "s2dex2"), ("gsps2dex", "s2dex"), ("gspturbo3d", "turbo3d"), ("gspzsort", "zsortp")):
        if b.startswith(pre):
            if fam is None:
                return None
            if "rej" in b:
                flags.append("rej")
                if fam == "f3dex2":
                    fam = "f3dex2_rej"
            if pre == "gspf3dlp":
                flags.append("lp")
            if pre == "gspf3dnon":
                flags.append("non")
            return fam, sorted(set(flags))
    return None


def elf_segments(elf_path):
    from elftools.elf.elffile import ELFFile
    out = {}
    with open(elf_path, "rb") as fh:
        elf = ELFFile(fh)
        syms = {}
        for sec in elf.iter_sections():
            if sec.header.sh_type == "SHT_SYMTAB":
                for s in sec.iter_symbols():
                    if SYM.match(s.name):
                        syms[s.name] = s["st_value"]
        secs = [s for s in elf.iter_sections() if s.header.sh_type == "SHT_PROGBITS" and s.header.sh_addr]
        for name, addr in syms.items():
            base, kind, edge = SYM.match(name).groups()
            if edge != "Start":
                continue
            end = syms.get(base + kind + "End")
            if end is None or end <= addr:
                continue
            for s in secs:
                a = s.header.sh_addr
                if a <= addr and end <= a + s.header.sh_size:
                    out[(base, kind.lower())] = s.data()[addr - a: end - a]
                    break
    return out


MAPLINE = re.compile(r"^\s+0x([0-9a-fA-F]{8,16})\s+(\w+)\s*$")


def map_segments(map_path, image):
    """GNU ld map: symbol vaddrs + section load addresses -> blobs cut from an uncompressed ROM image."""
    secv, secl = None, None
    syms = {}
    sec = re.compile(r"^(\S+)?\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)(?:\s+load address 0x([0-9a-fA-F]+))?")
    for line in open(map_path, encoding="utf8", errors="replace"):
        m = sec.match(line)
        if m and m.group(1) and m.group(1).startswith(".") and m.group(4):
            secv, secl = int(m.group(2), 16), int(m.group(4), 16)
            continue
        m = MAPLINE.match(line)
        if m and SYM.match(m.group(2)) and secv is not None:
            v = int(m.group(1), 16) & 0xFFFFFFFF
            syms[m.group(2)] = secl + (v - (secv & 0xFFFFFFFF))
    out = {}
    for name, rom in syms.items():
        base, kind, edge = SYM.match(name).groups()
        end = syms.get(base + kind + "End")
        if edge == "Start" and end and end > rom:
            out[(base, kind.lower())] = image[rom:end]
    return out


def dir_segments(path):
    """Decomp-extracted blobs: <name>.textbin.bin / <name>.databin.bin / <name>.text.bin / <name>.data.bin."""
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            m = re.match(r"^(\w+?)[._](text|data)(?:bin)?\.bin$", f, re.I)
            n = re.match(r"^(\w+?)(_data)?\.bin$", f, re.I)       # mk64 style: F3DEX.bin + F3DEX_data.bin
            if m:
                out[(m.group(1), m.group(2).lower())] = open(os.path.join(root, f), "rb").read()
            elif n and name_family(n.group(1)):
                out[(n.group(1), "data" if n.group(2) else "text")] = open(os.path.join(root, f), "rb").read()
    return out


def gfx_string_family(data):
    """Dirty side: family/version/flags from a gfx data segment's version string (never stored)."""
    m = re.search(rb"RSP (SW Version: [0-9.A-Z]+|Gfx ucode [ -~]{4,60})", data)
    if not m:
        return None
    s = m.group(0).decode("ascii", "replace")
    v = re.search(r"\b(\d)\.(\d\d)\b", s)
    ver = int(v.group(1)) * 100 + int(v.group(2)) if v else 0
    if "SW Version" in s:
        return "f3d", ver, []
    name = s[14:].split()[0] if len(s) > 14 else ""
    fl = [f for f in ("fifo", "xbus") if f in s.lower()]
    if ".NoN" in s:
        fl.append("non")
    if ".Rej" in s:
        fl.append("rej")
    if name.startswith("F3DZEX"):
        return ("f3dzex2_oot" if ver < 206 else "f3dzex2_mm"), ver, fl
    if name.startswith(("F3DLP",)):
        fl.append("lp")
    if name.startswith(("F3D", "F3DEX", "F3DLX", "F3DLP")):
        return ("f3dex2" + ("_rej" if "rej" in fl else "") if ver >= 200 else "f3dex"), ver, fl
    if name.startswith("L3D"):
        return ("l3dex2" if ver >= 200 else "l3dex"), ver, fl
    if name.startswith("S2DEX2") or (name.startswith("S2DEX") and ver >= 200):
        return "s2dex2", ver, fl
    if name.startswith("S2DEX"):
        return "s2dex", ver, fl
    if name.startswith("ZSort"):
        return "zsortp", ver, fl
    return None


def audio_family(data):
    if len(data) < 0x34:
        return None
    w = lambda o: struct.unpack_from(">I", data, o)[0]
    if w(0) == 1 and w(0x30) == 0xF0000F00:
        key = "abi1:%08x" % w(0x28)
    elif w(0) == 1:
        key = "abi2:%08x" % w(0x10)
    else:
        key = "abi3:%08x" % w(0x10)
    return AUDIO_SIG.get(hashlib.sha1(key.encode()).hexdigest()[:10])


def window(blob):
    """Offset of the first 8-aligned 64-byte window with >= 24 distinct byte values (skips zero runs)."""
    for w in range(0, max(1, len(blob) - 63), 8):
        if len(set(blob[w:w + 64])) >= 24:
            return w
    return 0


def short_hashes(blob):
    w = window(blob)
    return w, hashlib.sha1(blob[w:w + 64]).hexdigest()[:16], hashlib.sha1(blob).hexdigest()[:12]


# ------------------------------------------------------------------ known table + scan
def load_known():
    if os.path.exists(KNOWN):
        return json.load(open(KNOWN))
    return {"note": "short hashes of retail RSP microcode segments (no retail bytes)", "entries": []}


def scan_known(rom, known, step=8):
    win = {}
    for e in known["entries"]:
        win.setdefault(e["h64"], []).append(e)
    hits = []
    sha1 = hashlib.sha1
    for off in range(0, len(rom) - 64, step):
        h = sha1(rom[off:off + 64]).hexdigest()[:16]
        if h in win:
            for e in win[h]:
                start = off - e.get("w", 0)
                if start >= 0 and sha1(rom[start:start + e["size"]]).hexdigest()[:12] == e["h"]:
                    hits.append(dict(e, offset=start))
                    break
            else:
                hits.append({"family": win[h][0]["family"], "seg": win[h][0]["seg"], "offset": off,
                             "size": None, "partial": True})
    return hits


def heuristic(rom):
    """Count retail-looking microcode version strings (no text printed or stored)."""
    return len(re.findall(rb"RSP (?:Gfx ucode|SW Version)", rom))


def segments_for(path, elf=None, mapf=None, image=None):
    if os.path.isdir(path):
        return dir_segments(path)
    if elf:
        return elf_segments(elf)
    if mapf:
        return map_segments(mapf, image)
    if open(path, "rb").read(4) == b"\x7fELF":
        return elf_segments(path)
    return {}


def classify(base, kind, blob, override, data_blob=None):
    if base in override:
        return override[base], 0, []
    nf = name_family(base)
    fam, ver, fl = (nf[0], 0, nf[1]) if nf else (None, 0, [])
    probe = data_blob if data_blob is not None else (blob if kind == "data" else None)
    if probe is not None:
        g = gfx_string_family(probe)
        if g:
            fam = fam if fam in ("f3dex2_rej",) else g[0]
            ver, fl = g[1], sorted(set(fl) | set(g[2]))
        elif fam in ("aspmain", "naudio", None):
            fam = audio_family(probe) or fam
    return fam, ver, fl


# ------------------------------------------------------------------ commands
def cmd_stub(a):
    data = make_stub(a.family, a.seg, a.size, a.version, a.flags or 0)
    open(a.o, "wb").write(data)
    print("stub %s %s %d bytes -> %s" % (family_name(a.family), a.seg.upper(), a.size, a.o))


def write_stubs(spec, out):
    """spec: {"ucode": [{"name", "family", "version", "flags", "text", "data", "boot"?}]} -> files."""
    n = 0
    for e in spec["ucode"]:
        for seg in ("text", "data"):
            size = e.get(seg)
            if not size:
                continue
            kind = "boot" if family_name(e["family"]) == "boot" and seg == "text" else seg
            path = os.path.join(out, e.get("files", {}).get(seg, "%s.%sbin.bin" % (e["name"], seg)))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "wb").write(make_stub(e["family"], kind, size, e.get("version", 0), e.get("flags", [])))
            n += 1
    return n


def cmd_stubs(a):
    n = write_stubs(json.load(open(a.spec)), a.out)
    print("%d stub segments -> %s" % (n, a.out))


def cmd_scan(a):
    raw = open(a.rom, "rb").read()
    rom, order = to_z64(raw)
    known = load_known()
    hits = scan_known(rom, known)
    tags = find_tags(rom)
    print("%s (%s, %d KB): %d retail ucode segment(s) from ucode_known.json, %d tag stub(s), "
          "%d retail-looking ucode version string(s)" % (os.path.basename(a.rom), order, len(rom) // 1024,
                                                        len(hits), len(tags), heuristic(rom)))
    for h in hits:
        print("  RETAIL %-12s %-4s @0x%06x %s" % (h["family"], h["seg"], h["offset"],
                                                  "partial match" if h.get("partial") else "%d bytes" % h["size"]))
    for t in tags:
        print("  tag    %-12s %-4s @0x%06x %5d bytes v%d %s%s" % (t["family"], t["seg"], t["offset"], t["size"],
                                                               t["version"], ",".join(t["flags"]),
                                                               "" if t["crc_ok"] else " BAD-CRC"))
    if a.elf or a.map:
        segs = segments_for(a.rom, a.elf, a.map, rom)
        for (base, kind), blob in sorted(segs.items()):
            t = parse_tag(blob)
            print("  sym    %-24s %-4s %5d bytes %s" % (base, kind, len(blob),
                                                       ("tag " + t["family"]) if t else "NOT A TAG (retail?)"))
    return 1 if hits else 0


def cmd_learn(a):
    override = dict(x.split("=", 1) for x in a.family or [])
    image = to_z64(open(a.path, "rb").read())[0] if os.path.isfile(a.path) else None
    segs = segments_for(a.path, a.elf, a.map, image)
    known = load_known()
    have = {(e["h"], e["seg"]) for e in known["entries"]}
    added = 0
    for (base, kind), blob in sorted(segs.items()):
        fam, ver, fl = classify(base, kind, blob, override, segs.get((base, "data")))
        if not fam or not blob:
            print("  skip %s.%s (family unknown; pass --family %s=<family>)" % (base, kind, base))
            continue
        w, h64, h = short_hashes(blob)
        seg = "boot" if fam == "boot" else kind
        if (h, seg) in have:
            continue
        known["entries"].append({"family": fam, "seg": seg, "version": ver, "flags": fl, "size": len(blob),
                                 "w": w, "h64": h64, "h": h, "symbol": base})
        have.add((h, seg))
        added += 1
        print("  + %-12s %-4s v%-3d %-10s %5d bytes (%s)" % (fam, seg, ver, ",".join(fl), len(blob), base))
    known["entries"].sort(key=lambda e: (e["family"], e["seg"], e["version"], e["h"]))
    json.dump(known, open(KNOWN, "w"), indent=1)
    print("learn: %d segment(s) seen, %d new -> %s (%d entries)" % (len(segs), added, KNOWN, len(known["entries"])))


def cmd_swap(a):
    raw = open(a.rom, "rb").read()
    rom, order = to_z64(raw)
    out = bytearray(rom)
    spec = {e["name"]: e for e in json.load(open(a.spec))["ucode"]} if a.spec else {}
    jobs = []                                    # (offset, size, family, seg, version, flags)
    segs = segments_for(a.rom, a.elf, a.map, rom) if (a.elf or a.map) else {}
    for (base, kind), blob in segs.items():
        if parse_tag(blob):
            continue
        off = rom.find(blob)
        if off < 0 or rom.find(blob, off + 1) >= 0:
            print("  %s.%s: not found once in the ROM (compressed segment?) - rebuild with `stubs` instead"
                  % (base, kind))
            continue
        e = spec.get(base, {})
        fam, ver, fl = (e["family"], e.get("version", 0), e.get("flags", [])) if e else \
            classify(base, kind, blob, {}, segs.get((base, "data")))
        jobs.append((off, len(blob), fam, "boot" if fam == "boot" else kind, ver, fl))
    if not segs:
        for h in scan_known(rom, load_known()):
            if not h.get("partial"):
                jobs.append((h["offset"], h["size"], h["family"], h["seg"], h["version"], h["flags"]))
    for off, size, fam, seg, ver, fl in sorted(jobs):
        out[off:off + size] = make_stub(fam, seg, size, ver, fl)
        print("  swap %-12s %-4s @0x%06x %5d bytes" % (fam, seg, off, size))
    new = bytes(out)
    cic = None
    if a.crc == "auto":
        new, cic = fix_crc(rom, new)
    open(a.o, "wb").write(from_z64(new, order))
    left = scan_known(to_z64(new)[0], load_known())
    print("swap: %d segment(s) replaced, crc %s, %d known retail segment(s) left -> %s"
          % (len(jobs), cic or a.crc, len([h for h in left if not h.get("partial")]), a.o))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cleanroom.ucode", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stub")
    p.add_argument("family"); p.add_argument("seg"); p.add_argument("size", type=int)
    p.add_argument("-o", required=True); p.add_argument("--version", default=0); p.add_argument("--flags")
    p.set_defaults(fn=cmd_stub)
    p = sub.add_parser("stubs")
    p.add_argument("spec"); p.add_argument("out"); p.set_defaults(fn=cmd_stubs)
    for name, fn in (("scan", cmd_scan), ("learn", cmd_learn), ("swap", cmd_swap)):
        p = sub.add_parser(name)
        p.add_argument("rom" if name != "learn" else "path")
        p.add_argument("--elf"); p.add_argument("--map")
        if name == "learn":
            p.add_argument("--family", action="append", help="symbol=family override")
        if name == "swap":
            p.add_argument("-o", required=True); p.add_argument("--spec"); p.add_argument("--crc", default="auto")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
