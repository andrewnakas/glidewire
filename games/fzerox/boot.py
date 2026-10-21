"""Boot pieces with no retail program bytes: our IPL3 stub, CRUCODE1 microcode tags, our ROM header.

    apply(out)   writes bin/us/rev0/boot.bin (IPL3 slot), bin/us/rev0/rsp/*.bin (tag stubs) and
                 asm/us/rev0/header.s (title GLIDEWIRE, cart id GW) into the clean tree

IPL3: games/fzerox/ipl3/ipl3.s (our CC0 stub, shared with the Tonic Tumble rebuild): copies 1 MiB from cart
0x1000 to the header entry point, fills the libultra boot globals, jumps. It applies no CIC-specific entry
offset, so the header carries the real entry point (0x80067000, the decomp's `entry` segment vram).
Emulators boot it; real consoles need a CIC-signed IPL3.
Microcode: spec/ucode.json (family/version/flags/sizes) -> cleanroom.ucode.write_stubs. HLE emulator cores
patched for the tags are required (n64cleanrecomp ports/emu/ucode-tags).
"""
import json
import os
import shutil
import subprocess

from cleanroom import ucode

HERE = os.path.dirname(os.path.abspath(__file__))
VER = "us/rev0"
IPL3_SLOT = 0xFC0

ENTRY = 0x80067000
TITLE = "GLIDEWIRE"
CART_ID = "GW"

HEADER = """.section .data

.word 0x80371240       /* PI BSB Domain 1 register */
.word 0x0000000F       /* Clockrate setting */
.word 0x{entry:08X}       /* Entrypoint address (real address: our IPL3 adds no CIC offset) */
.word 0x00001449       /* Revision */
.word 0x00000000       /* Checksum 1 (fixcrc.py) */
.word 0x00000000       /* Checksum 2 (fixcrc.py) */
.word 0x00000000       /* Unknown 1 */
.word 0x00000000       /* Unknown 2 */
.ascii "{title}" /* Internal name */
.word 0x00000000       /* Unknown 3 */
.word 0x0000004E       /* Media: N */
.ascii "{cart}"            /* Cartridge ID */
.ascii "E"             /* Country code */
.byte 0x00             /* Version */
"""


def anchors_spec(pristine, baserom, out=os.path.join(HERE, "spec", "ipl3_anchors.json")):
    """DIRTY, once: the decomp code reads bit fields of ROM words inside the IPL3 slot as gameplay constants
    (racer.c machine tuning, main_menu.c / machine_draw.c anti-piracy checks). Record, per word, the union of the
    bits the code uses and the value of only those bits (code-hard-wired anchors; the rest of the word is unused)."""
    import re
    masks = {}
    for dp, _, fn in os.walk(os.path.join(pristine, "src")):
        for f in fn:
            if not f.endswith(".c"):
                continue
            t = open(os.path.join(dp, f), encoding="utf8", errors="replace").read()
            for m in re.finditer(r"ROM_READ\((0x[0-9A-Fa-f]+)\)", t):
                a = int(m.group(1), 16)
                tail = t[m.end():m.end() + 40]
                f2 = re.match(r"\s*>>\s*(0x[0-9A-Fa-f]+|\d+)\)\s*&\s*(0x[0-9A-Fa-f]+)", tail)
                if f2:
                    mk = int(f2.group(2), 16) << int(f2.group(1), 0)
                elif a == 0x6E8:
                    mk = 0xFFFF                   # main_menu.c: (word & 0xFFFF) - 0x997A
                elif a == 0x510 and "racer" not in f:
                    mk = 1                        # machine_draw.c: (word & 1)
                else:
                    mk = 0xFFFFFFFF
                masks[a] = masks.get(a, 0) | (mk & 0xFFFFFFFF)
    rom = open(baserom, "rb").read()
    spec = {f"0x{a:03X}": [m, int.from_bytes(rom[a:a + 4], "big") & m] for a, m in sorted(masks.items())}
    json.dump({"note": "IPL3-slot ROM words the game code reads as constants: [bit mask used, value of those bits]",
               "anchors": spec}, open(out, "w"), indent=0)
    print(f"ipl3 anchors: {len(spec)} words, {sum(bin(m).count('1') for m, _ in spec.values())} bits")


def write_ipl3(path):
    pre = os.environ.get("MIPS_PREFIX", r"C:/Users/andre/.local/mips64/bin/mips64-elf-")
    tmp = path + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(HERE, "ipl3", "ipl3.s")
    subprocess.run([pre + "gcc", "-c", "-march=vr4300", "-mabi=32", "-EB", "-mno-abicalls", "-fno-PIC", "-G0",
                    src, "-o", os.path.join(tmp, "ipl3.o")], check=True)
    subprocess.run([pre + "ld", "-EB", "-Ttext=0xA4000040", "-e", "ipl3_start", "-o", os.path.join(tmp, "ipl3.elf"),
                    os.path.join(tmp, "ipl3.o")], check=True)
    subprocess.run([pre + "objcopy", "-O", "binary", "-j", ".text", os.path.join(tmp, "ipl3.elf"),
                    os.path.join(tmp, "ipl3.bin")], check=True)
    code = open(os.path.join(tmp, "ipl3.bin"), "rb").read()
    assert len(code) <= IPL3_SLOT
    blob = bytearray(code.ljust(IPL3_SLOT, b"\0"))
    # Code anchors: the game reads bit fields of 33 words in this slot as constants (racer.c machine tuning,
    # main_menu.c 0x6E8 & 0xFFFF == 0x997A, machine_draw.c 0x510 bit 0). Only those bits are set (spec/
    # ipl3_anchors.json, 320 bits); everything else in the slot is our code or zero. Our code sits at ROM 0xA40.
    anc = json.load(open(os.path.join(HERE, "spec", "ipl3_anchors.json")))["anchors"]
    for a, (mk, val) in anc.items():
        off = int(a, 16) - 0x40
        word = int.from_bytes(blob[off:off + 4], "big")
        assert word & mk == 0 or word & mk == val, f"IPL3 code collides with anchor {a}"
        blob[off:off + 4] = ((word & ~mk & 0xFFFFFFFF) | val).to_bytes(4, "big")
    open(path, "wb").write(bytes(blob))
    shutil.rmtree(tmp)
    return len(code)


def apply(out):
    b = os.path.join(out, "bin", VER)
    os.makedirs(b, exist_ok=True)
    n_ipl3 = write_ipl3(os.path.join(b, "boot.bin"))
    # libleo font-address lookups (64DD only, never reached on a cartridge): our stub returning 0, padded to
    # the slot size (structure), instead of the retail binary blobs
    for name, size in (("getaadr", 9376), ("getkadr", 2880)):
        p = os.path.join(b, "leo", "lib", name + ".textbin.bin")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(bytes.fromhex("03E0000800001025").ljust(size, b"\0"))
    spec = json.load(open(os.path.join(HERE, "spec", "ucode.json")))
    n = ucode.write_stubs(spec, b)
    with open(os.path.join(out, "asm", VER, "header.s"), "w", newline="\n") as f:
        f.write(HEADER.format(entry=ENTRY, title=TITLE.ljust(20)[:20], cart=CART_ID))
    print(f"boot: our IPL3 ({n_ipl3} B code), {n} CRUCODE1 microcode stubs, header '{TITLE}' {CART_ID}E")
