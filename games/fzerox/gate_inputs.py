"""Inputs for the originality gate (written under one gate dir; the retail half stays on the dirty side).

    python -m games.fzerox.gate_inputs <dirty tree> <clean tree> <gate dir>

code/     retail_code.bin + clean_code.bin: every executable output section of the two linked ELFs (boot code,
          main, libleo, overlays), in name order with zero gaps; compiled/*.bin = the linked bytes of each
          object our build compiled or assembled from decomp sources (linker map), so the check proves the
          code region is compiler/assembler output. Objects that only .incbin a binary blob are NOT listed.
text/     retail strings (decomp source literals, texture-label transcriptions, old voice lines) vs ours
geo/      course ribbons retail vs ours (+ machgeo meshes when that module exists)
music/    decoded music slots retail vs ours (audio.py gate)
images/   retail PNGs (dirty_png) vs our textures (generate.py writes clean/gate/images)
"""
import json
import os
import re
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VER = "us/rev0"
NOT_COMPILED = ()   # every blob in our build is ours now (IPL3 stub, CRUCODE1 tags, libleo stubs)


def elf_sections(path):
    d = open(path, "rb").read()
    shoff, = struct.unpack(">I", d[0x20:0x24])
    shentsize, shnum, shstrndx = struct.unpack(">HHH", d[0x2E:0x34])
    secs = []
    for i in range(shnum):
        o = shoff + i * shentsize
        name, typ, flags, addr, off, size = struct.unpack(">IIIIII", d[o:o + 24])
        secs.append([name, typ, flags, addr, off, size])
    so = secs[shstrndx][4]
    out = {}
    for name, typ, flags, addr, off, size in secs:
        nm = d[so + name:d.index(b"\0", so + name)].decode()
        if typ == 1 and flags & 4 and addr >= 0x80000000:          # PROGBITS + EXECINSTR
            out[nm] = (addr, d[off:off + size])
    return out


def map_objects(path):
    """[(output section, vma, size, object)] for .text/.data/.rodata input sections."""
    cur, out = None, []
    for line in open(path, encoding="utf8", errors="replace"):
        m = re.match(r"^(\.\w+)\s+0x([0-9a-f]+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^ \.(text|data|rodata|rdata)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+) (\S+\.o)", line)
        if m and cur:
            out.append((cur, int(m.group(2), 16), int(m.group(3), 16), m.group(4)))
    return out


def code(dirty, clean, gate):
    g = os.path.join(gate, "code")
    shutil.rmtree(g, ignore_errors=True)
    os.makedirs(os.path.join(g, "compiled"))
    b = os.path.join(VER.replace("/", os.sep))
    elf = lambda t: os.path.join(t, "build", VER, "fzerox.us.rev0.elf")
    R, C = elf_sections(elf(dirty)), elf_sections(elf(clean))
    names = sorted(set(R) & set(C))
    gap = bytes(512)
    open(os.path.join(g, "retail_code.bin"), "wb").write(gap.join(R[n][1] for n in names))
    open(os.path.join(g, "clean_code.bin"), "wb").write((bytes([0xA5]) * 512).join(C[n][1] for n in names))
    # link-order runs: consecutive slices of compiled objects (alignment padding between them included) form one
    # run; a blob object (incbin) ends the run, so its bytes are never "explained"
    skipped = 0
    slices = []
    for sec, vma, size, obj in map_objects(os.path.join(clean, "build", VER, "fzerox.us.rev0.map")):
        if sec not in C or size == 0:
            continue
        blob = any(x in obj for x in NOT_COMPILED)
        skipped += blob
        slices.append((sec, vma, size, blob))
    slices.sort()
    runs = []
    for sec, vma, size, blob in slices:
        if blob:
            runs.append(None)
            continue
        if runs and runs[-1] and runs[-1][0] == sec and 0 <= vma - runs[-1][2] <= 4096:
            runs[-1][2] = vma + size
        else:
            runs.append([sec, vma, vma + size])
    n_obj = 0
    for r in runs:
        if not r:
            continue
        base, data = C[r[0]]
        a, b = r[1] - base, r[2] - base
        if 0 <= a < len(data):
            open(os.path.join(g, "compiled", f"{n_obj:05d}.bin"), "wb").write(data[a:b])
            n_obj += 1
    print(f"gate code: {len(names)} executable sections, {n_obj} link-order runs of compiled objects, {skipped} "
          f"blob objects left out")


def text(dirty_gate, clean, gate):
    t = os.path.join(gate, "text")
    os.makedirs(t, exist_ok=True)
    retail = list(json.load(open(os.path.join(dirty_gate, "text_retail.json"), encoding="utf8")).values())
    lab = json.load(open(os.path.join(dirty_gate, "labels_retail.json"), encoding="utf8"))
    retail += [v["text"] for k, v in lab.items() if not k.startswith("_")]
    vt = os.path.join(dirty_gate, "voices_retail.json")
    if os.path.exists(vt):
        retail += json.load(open(vt, encoding="utf8"))
    json.dump(retail, open(os.path.join(t, "retail.json"), "w", encoding="utf8"), ensure_ascii=False)
    ours = json.load(open(os.path.join(clean, "gate", "text_clean_src.json"), encoding="utf8"))
    lab = json.load(open(os.path.join(HERE, "text_labels.json"), encoding="utf8"))
    ours += [v["text"] for k, v in lab.items() if not k.startswith("_")]
    vl = json.load(open(os.path.join(HERE, "voice_lines.json"), encoding="utf8"))
    ours += [v["text"] for k, v in vl.items() if not k.startswith("_")]
    json.dump(ours, open(os.path.join(t, "clean.json"), "w", encoding="utf8"), ensure_ascii=False)
    print(f"gate text: {len(retail)} retail strings, {len(ours)} ours")


def geo(dirty, clean, gate):
    from .courses import parse, ribbon, save_mesh
    g = os.path.join(gate, "geo")
    cs = parse(os.path.join(dirty, "src", "assets", VER, "course_data", "course_data.c"),
               os.path.join(dirty, "include"))
    for c in cs:
        v, t = ribbon(c)
        save_mesh(os.path.join(g, "retail_courses", c["name"] + ".json"), v, t)
    src = os.path.join(clean, "gate", "courses")
    dst = os.path.join(g, "clean_courses")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    print(f"gate geo: {len(cs)} course ribbons each side")


def images(dirty_png, clean, gate):
    for side, src in (("retail_images", dirty_png), ("clean_images", os.path.join(clean, "gate", "images"))):
        dst = os.path.join(gate, side)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    print("gate images: copied")


def main(argv):
    dirty, clean, gate = argv[1:4]
    os.makedirs(gate, exist_ok=True)
    code(dirty, clean, gate)
    text(gate, clean, gate)
    geo(dirty, clean, gate)
    images(os.path.join(os.path.dirname(dirty), "dirty_png"), clean, gate)
    from . import sfxseq
    rel = os.path.join("src", "assets", VER, "audio_seq", "audio_seq.c")
    for side, tree in (("retail", dirty), ("clean", clean)):
        n = sfxseq.to_midi(os.path.join(tree, rel), os.path.join(gate, "seq", side, "audio_seq.mid"))
        print(f"gate seq: {side} {n} notes")
    from .audio import gate as audio_gate
    audio_gate(dirty, clean, os.path.join(gate, "music"))
    try:
        from . import machgeo
        machgeo.export_gate(dirty, clean, os.path.join(gate, "geo"))
    except ImportError:
        print("gate geo: machgeo not present")


if __name__ == "__main__":
    main(sys.argv)
