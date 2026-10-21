"""Taint report: generated F-Zero X assets vs the retail extraction (dev check, dirty tree needed).

Scanned (clean vs every retail stream, cleanroom.taint windows, runs >= FAIL_RUN fail):
  * every texture/TLUT as stored bytes (MIO0 textures: compressed and decompressed) and, for non-CI formats,
    decoded RGBA,
  * every sample: its VADPCM bytes in audio_table.bin, its decoded PCM, its codebook (audio_bank.c),
  * the 38 1-bit masks and the crash-screen font the decomp keeps in C.
Kept facts (sequences, soundfont structure, course data, ghosts, code, geometry) are listed, not scanned.

    python -m games.fzerox.taint_report <dirty tree> <clean tree>
"""
import json
import os
import re
import sys

import crunch64
import numpy as np

from cleanroom import taint
from cleanroom.audio import vadpcm
from cleanroom.gfx import texfmt

from .audio import BANK_C, parse_bank
from .extract_spec import CSIZE, FMTS, VER, inc_bytes
from .masks import RX as MASK_RX

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec")


def c_words(path, name):
    s = open(path).read()
    m = re.search(r"\b%s\[[^\]]*\]\s*=\s*\{(.*?)\n\};" % re.escape(name), s, re.S)
    return b"".join(int(v, 16).to_bytes(4, "big") for v in re.findall(r"0x[0-9A-Fa-f]+", m.group(1)))


def audio_streams(tree, aud):
    table = open(os.path.join(tree, "bin", VER, "audio_table.bin"), "rb").read()
    samples, books = parse_bank(open(os.path.join(tree, BANK_C)).read())
    bases = aud["bases"]
    for name, d in aud["samples"].items():
        st = bases[d["bank"]][0] + d["addr"]
        data = table[st:st + d["size"]]
        yield "adpcm:" + name, data
        _, order, npred, coefs = books[(d["font"], d["book"])]
        yield "book:" + name, np.asarray(coefs, ">i2").tobytes()
        pcm = vadpcm.decode(data, {"order": order, "npred": npred, "book": coefs}, d["n"])
        yield "pcm:" + name, pcm.astype(">i2").tobytes()


def streams(tree, tex, aud):
    for sym, e in tex.items():
        p = os.path.join(tree, e["file"])
        if e.get("mio0"):
            raw = inc_bytes(p, 1)
            # MIO0 = header, layout flag bits, back-reference commands, literal bytes. Only the literals carry
            # pixel bytes; flags/commands are the encoder's (identical for any blank run in any picture), and
            # the decoded texels are scanned below.
            lit = int.from_bytes(raw[12:16], "big")
            yield "mio0lit:" + sym, raw[lit:]
            b = crunch64.mio0.decompress(raw)
        else:
            b = inc_bytes(p, CSIZE[e["ctype"]])
        yield "tex:" + sym, b
        if e["fmt"] != "TLUT" and "w" in e:
            f, s = FMTS[e["fmt"]]
            if f != texfmt.CI:
                img = texfmt.decode(b, e["w"], e["h"], f, s)
                yield "rgba:" + sym, img[img[..., 3] > 0].tobytes()
    if aud:
        yield from audio_streams(tree, aud)
    cac = open(os.path.join(tree, "src", "assets", VER, "common_assets_compressed", "common_assets_compressed.c")).read()
    for m in MASK_RX.finditer(cac):
        vals = [int(v) for v in re.findall(r"\d+", m.group(2))]
        if len(vals) == 512:
            yield "mask:" + m.group(1), bytes(vals)
    yield "code:sys_fault", c_words(os.path.join(tree, "src/sys/sys_fault.c"), "sFaultCharPixelFlags")


def blank_compiled_code(clean, rom):
    """Zero the ROM bytes of every object compiled from decomp sources (src/**, asm/**; not assets, bins or the
    microcode tags), located by their linked bytes: pass 3 then looks only at data and blobs."""
    from .gate_inputs import elf_sections, map_objects
    C = elf_sections(os.path.join(clean, "build", VER, "fzerox.us.rev0.elf"))
    out = bytearray(rom)
    n = 0
    for sec, vma, size, obj in map_objects(os.path.join(clean, "build", VER, "fzerox.us.rev0.map")):
        if sec not in C or size < 64 or "/src/assets/" in obj or "/bin/" in obj or "/src/rsp/" in obj:
            continue
        base, data = C[sec]
        piece = data[vma - base:vma - base + size]
        j = rom.find(piece)
        if j >= 0:
            out[j:j + size] = bytes(size)
            n += 1
    return bytes(out), n


def retail_blobs(dirty):
    """Retail bytes of the data the 2026-10 rebuild replaced (dirty side only)."""
    rom = open(os.path.join(dirty, "build", VER, "fzerox_uncompressed.us.rev0.z64"), "rb").read()
    txt = open(os.path.join(dirty, "build", VER, "fzerox.us.rev0.map"), encoding="utf8", errors="replace").read()
    for seg in ("course_data", "staff_ghost_records"):
        a = re.search(r"0x0*([0-9a-f]+)\s+%s_ROM_START\b" % seg, txt)
        b = re.search(r"0x0*([0-9a-f]+)\s+%s_ROM_END\b" % seg, txt)
        if a and b:
            yield "retail:" + seg, rom[int(a.group(1), 16):int(b.group(1), 16)]
        else:
            print(f"  (no ROM range for {seg} in the dirty map)")
    for f in ("boot.bin", "leo/lib/getaadr.textbin.bin", "leo/lib/getkadr.textbin.bin"):
        yield "retail:" + f, open(os.path.join(dirty, "bin", VER, f), "rb").read()
    rd = os.path.join(dirty, "bin", VER, "rsp")
    for f in sorted(os.listdir(rd)):
        yield "retail:rsp/" + f, open(os.path.join(rd, f), "rb").read()
    cac = open(os.path.join(dirty, "src", "assets", VER, "common_assets_compressed", "common_assets_compressed.c")).read()
    for m in re.finditer(r"s16 (D_F26[0-9A-F]+)\[\] = \{([^}]*)\}", cac):
        vals = [int(v) for v in m.group(2).split(",") if v.strip()]
        yield "retail:cpu:" + m.group(1), np.asarray(vals, ">i2").tobytes()


def main(argv):
    dirty, clean = argv[1], argv[2]
    tex = json.load(open(os.path.join(SPEC, "textures.json"), encoding="utf-8"))
    aud = json.load(open(os.path.join(SPEC, "audio.json")))
    if "--only" in argv:   # quick re-check of some textures against ALL retail textures (no audio in the clean set)
        rx = re.compile(argv[argv.index("--only") + 1])
        index = taint.build_index(s for _, s in streams(dirty, tex, None))
        sub = {k: v for k, v in tex.items() if rx.search(k) or rx.search(v["yaml"])}
        hits = taint.scan(index, (x for x in streams(clean, sub, None) if not x[0].startswith(("mask:", "code:"))))
        bad = sorted((h for h in hits if h[3] >= taint.FAIL_RUN), key=lambda h: -h[3])
        print(f"subset {len(sub)} textures: {len(bad)} failing")
        for label, off, nw, run in bad[:10]:
            print(f"  FAIL {label} run {run} B")
        return 1 if bad else 0
    # two passes (pictures + code, then audio) keep the index small enough for a busy machine
    index = taint.build_index(s for _, s in streams(dirty, tex, None))
    hits = taint.scan(index, streams(clean, tex, None))
    del index
    index = taint.build_index(s for _, s in audio_streams(dirty, aud))
    hits += taint.scan(index, audio_streams(clean, aud))
    del index
    # pass 3 (2026-10 rebuild): retail data that used to be kept must not appear anywhere in our ROM image
    index = taint.build_index(s for _, s in retail_blobs(dirty))
    rom = open(os.path.join(clean, "build", VER, "fzerox_uncompressed.us.rev0.z64"), "rb").read()
    rom, n_code = blank_compiled_code(clean, rom)
    hits += taint.scan(index, [("rom:uncompressed", rom)])
    print(f"pass 3: {n_code} slices of game code/data compiled from decomp C/asm blanked before the ROM scan "
          f"(e.g. leo_fault.c carries a font table that libleo's getaadr blob also contains)")
    bad = sorted((h for h in hits if h[3] >= taint.FAIL_RUN), key=lambda h: -h[3])
    print(f"scanned {len(tex)} textures/TLUTs (+MIO0 literals, +decoded), {len(aud['samples'])} samples (ADPCM, PCM, books), "
          f"38 masks, fault font; the whole clean ROM vs retail course data, staff ghosts, CPU guidance, RSP "
          f"microcode, IPL3, libleo blobs; {len(hits)} streams with short coincidental matches; "
          f"{len(bad)} failing (run >= {taint.FAIL_RUN} B)")
    print("structure kept, not scanned: soundfont layout (books regenerated), audio_seq sound-effect sequences, "
          "display-list topology (machgeo), game code (compiled from the decomp; see ORIGINALITY.md code check)")
    for label, off, nw, run in bad[:15]:
        print(f"  FAIL {label} run {run} B ({nw} windows)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
