"""CLEAN ROOM: build an fzerox decomp tree whose extracted assets are all generated.

    python -m games.fzerox.generate <pristine> <dirty-code-only inputs> <clean tree out>

Inputs:
  * pristine decomp source (no extracted assets),
  * from the dirty tree ONLY code/geometry that the user scope keeps: splat's asm/ (libultra asm, boot/RSP
    code bins), Torch's src/assets/*/*.c (display lists, vertices, course data, ghost inputs, CPU racing
    lines, note sequences) and include/assets headers, the linker scripts, asm-processor,
  * games/fzerox/spec (coarse texture facts) + our own drawings (labels, faces, icons).
Writes every Torch texture .inc.c / .incbin.c (MIO0, compressed here) and TLUT from the spec, and the 1-bit
ending masks inline in common_assets_compressed.c; never copies retail pixel data.
Audio samples (audio_table.bin) and codebooks (audio_bank.c) come from games.fzerox.audio.
"""
import json
import os
import re
import shutil
import sys

import crunch64
import numpy as np
from scipy.cluster.vq import kmeans2

from cleanroom.decomp.gen import from_digest, h32
from cleanroom.gfx import texfmt

from .extract_spec import FMTS, CSIZE, VER

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec")

# copied from the dirty tree: code and geometry only (never *.inc.c / *.incbin.c pixel data)
KEEP_DIRS = ["asm", "linker_scripts", "include/assets", "tools/asm-processor"]
END_PAD = 624384 - 0x40000      # F67900.bin: the 0xFF tail pad, 256 KB shorter so our (bigger) MIO0 data still fits 16 MB
KEEP_BIN = []   # libleo getaadr/getkadr: our stubs (boot.py)
# IPL3, RSP microcode and the ROM header are written by boot.py (our stub, CRUCODE1 tags, our header)
# u8[512] arrays in common_assets_compressed: 64x64 1-bit masks (ending fireworks shapes)
MASK_RX = re.compile(r"(u8 (D_F2[0-9A-F]{5})\[\] = \{)([^}]*)(\};)")


def copy_code(dirty, out):
    for d in KEEP_DIRS:
        src = os.path.join(dirty, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, d), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for b in KEEP_BIN:
        dst = os.path.join(out, "bin", VER, b)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(dirty, "bin", VER, b), dst)
    open(os.path.join(out, "bin", VER, "F67900.bin"), "wb").write(bytes([0xFF]) * END_PAD)
    n = 0
    adir = os.path.join(dirty, "src", "assets", VER)
    for d in os.listdir(adir):
        if not os.path.isdir(os.path.join(adir, d)):
            continue
        for f in os.listdir(os.path.join(adir, d)):
            if f.endswith(".c") and not f.endswith((".inc.c", ".incbin.c")) and f != "audio_bank.c":
                os.makedirs(os.path.join(out, "src", "assets", VER, d), exist_ok=True)
                shutil.copy2(os.path.join(adir, d, f), os.path.join(out, "src", "assets", VER, d, f))
                n += 1
    print(f"code/geometry copied: {n} asset .c files, {len(KEEP_BIN)} bins, {KEEP_DIRS}")


# ----------------------------------------------------------------- pictures

WRAP = {}    # symbol -> (wrap_s, wrap_t) from the kept display lists (G_TX_WRAP without mirror)
# segments the ROM build MIO0-compresses whole (tools/compress.py compressedEntries)
SEG_MIO0 = {"course_track_gfx", "machine_models", "mute_city_textures", "port_town_textures", "big_blue_textures",
            "sand_ocean_textures", "devils_forest_textures", "white_land_textures", "sector_textures",
            "red_canyon_textures", "fire_field_textures", "silence_textures", "ending_venue_textures", "podium_gfx"}
# ordered-dither strength per compressed segment, the most each can take and still compress under its retail size
QSTEP = {"podium_gfx": 32.0}   # ordered-dither level spacing per compressed segment (default 16 = 4-bit levels)
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - 0.47


def wrap_modes(tree):
    rx = re.compile(r"gsDPLoad(?:Texture|MultiBlock)\w*\(\s*(\w+),[^;]*?G_TX_(NO)?MIRROR \| G_TX_(WRAP|CLAMP),"
                    r"\s*G_TX_(NO)?MIRROR \| G_TX_(WRAP|CLAMP)")
    out = {}
    adir = os.path.join(tree, "src", "assets", VER)
    for d in os.listdir(adir):
        f = os.path.join(adir, d, d + ".c")
        if os.path.exists(f):
            for m in rx.finditer(open(f).read()):
                ws = m.group(2) == "NO" and m.group(3) == "WRAP"
                wt = m.group(4) == "NO" and m.group(5) == "WRAP"
                a, b = out.get(m.group(1), (False, False))
                out[m.group(1)] = (a or ws, b or wt)
    return out


def upsample(grid, n, w, h, ws, wt):
    """Bilinear grid -> w x h; periodic along wrapped axes so tiles join without seams."""
    g = np.asarray(grid, np.float32).reshape(n, n, 4)

    def axis(size, wrap):
        c = (np.arange(size, dtype=np.float32) + 0.5) / size * n - 0.5
        i0 = np.floor(c).astype(int)
        f = c - i0
        if wrap:
            return i0 % n, (i0 + 1) % n, f
        return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), np.clip(f, 0, 1) * (c > 0) * (c < n - 1) + (c >= n - 1)

    y0, y1, fy = axis(h, wt)
    x0, x1, fx = axis(w, ws)
    fy, fx = fy[:, None, None], fx[None, :, None]
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
    bot = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


HOOKS = []   # functions (sym, e) -> rgba or None, tried in order (labels, faces, icons ...)
HOOKED = [0]


# Our second design for slots whose first design came out too close to a retail picture (originality gate):
# tiny glyph cells where a plain letter shape is all there is, and one nearly empty 8x8 mask.
TWEAK = {"aFont4YTex": "shear", "aFont1PrimeTex": "shear", "aFont2PrimeTex": "dots", "aFont2TTex": "shear", "D_F2207C8": "dots"}


def _tweak(img, how):
    img = img.copy()
    h, w = img.shape[:2]
    if how == "shear":            # italic slant: our glyph leans right
        out = np.zeros_like(img)
        for y in range(h):
            s = int(round((h - 1 - y) * 0.35))
            out[y, s:] = img[y, :w - s] if s else img[y]
        return out
    yy, xx = np.mgrid[0:h, 0:w]   # "dots": a sparse diagonal dot pattern
    m = ((xx + yy) % 3 == 0)
    img[..., :3] = 255
    img[..., 3] = np.where(m, 255, 0)
    return img


def picture(sym, e):
    img = _picture(sym, e)
    if sym in TWEAK:
        img = _tweak(img, TWEAK[sym])
    if e["fmt"] in ("RGBA16", "RGBA32", "CI4", "CI8"):
        # opaque texels never exactly black (one 5-bit step up, invisible): flat black runs with an edge texel
        # otherwise coincide with retail black backgrounds (taint scan)
        img = img.copy()
        dark = (img[..., 3] >= 128)[..., None] & (img[..., :3] < 8)
        img[..., :3] = np.where(dark, 8, img[..., :3])
    return img


def _picture(sym, e):
    for hook in HOOKS:
        img = hook(sym, e)
        if img is not None:
            HOOKED[0] += 1
            if e["fmt"] in ("I4", "I8"):
                # intensity formats have no separate alpha: the shape must live in the intensity
                a = img[..., 3:4].astype(np.float32) / 255.0
                img = img.copy()
                img[..., :3] = np.clip(img[..., :3] * a, 0, 255).astype(np.uint8)
            return img
    packed = e.get("mio0") or e.get("yaml") in SEG_MIO0
    # Glidewire's own texture kit (theme.py): pattern + shape from structure and our palettes only
    from . import theme
    img = theme.texture(sym, e).astype(np.float32)
    if packed:
        # MIO0-stored: ordered dithering onto 4-bit levels (16 apart) with a per-texture Bayer phase and a
        # per-texture half-level channel offset. Few distinct values keep the segments compressible like the retail
        # art; the offset levels never sit on the retail 5-bit ramp values (taint scan).
        rng = np.random.default_rng(h32("bayer", sym))
        b = np.roll(BAYER + 0.47, tuple(rng.integers(0, 4, 2)), (0, 1))          # 0..1
        by = np.tile(b, (e["h"] // 4 + 1, e["w"] // 4 + 1))[:e["h"], :e["w"]]
        q = QSTEP.get(e.get("yaml"), 16.0)
        off = rng.choice([4.0, 12.0], 3)
        img[..., :3] = np.floor((img[..., :3] - off) / q + by[..., None]) * q + off
    else:
        # our own per-texel dither (+-1.5 steps of RGBA16): smooth gradients otherwise quantise into
        # the same texel runs as the retail art (taint scan)
        rng = np.random.default_rng(h32("dither", sym))
        img[..., :3] += rng.uniform(-12.0, 12.0, img.shape[:2] + (1,)) + rng.uniform(-4.0, 4.0, img.shape[:2] + (3,))
    return np.clip(img, 0, 255).astype(np.uint8)


def fmt_inc(data, ctype):
    cs = CSIZE[ctype]
    vals = [int.from_bytes(data[i:i + cs], "big") for i in range(0, len(data), cs)]
    per = {1: 16, 2: 8, 4: 4, 8: 2}[cs]
    w = 2 * cs
    lines = [", ".join(f"0x{v:0{w}X}" for v in vals[i:i + per]) + ", " for i in range(0, len(vals), per)]
    return "\n".join(lines) + "\n"


def build_palette(imgs, n, want_alpha):
    """Our own palette for all pictures sharing one TLUT: k-means over their opaque pixels."""
    px = np.concatenate([im.reshape(-1, 4) for im in imgs]).astype(np.float32)
    opaque = px[px[:, 3] >= 128][:, :3]
    k = n - 1 if want_alpha else n
    pal = np.zeros((n, 4), np.uint8)
    if len(opaque):
        rng = np.random.default_rng(h32("pal", n, len(opaque)))
        sample = opaque[rng.choice(len(opaque), min(len(opaque), 4096), replace=False)]
        uniq = np.unique(sample.astype(np.uint8), axis=0).astype(np.float32)
        kk = max(1, min(k, len(uniq)))
        cent, _ = kmeans2(sample, uniq[rng.choice(len(uniq), kk, replace=False)], minit="matrix", iter=12, seed=1)
        cent = np.clip(cent, 0, 255)
        off = 1 if want_alpha else 0
        pal[off:off + kk, :3] = cent.astype(np.uint8)
        pal[off:off + kk, 3] = 255
        pal[off + kk:, :] = pal[off + kk - 1]
    return pal


def index_image(img, pal, has_alpha):
    rgb = img[..., :3].reshape(-1, 1, 3).astype(np.float32)
    cand = pal[1:] if has_alpha else pal
    d = ((rgb - cand[None, :, :3].astype(np.float32)) ** 2).sum(-1)
    idx = d.argmin(1) + (1 if has_alpha else 0)
    if has_alpha:
        idx[img[..., 3].reshape(-1) < 128] = 0
    return idx.reshape(img.shape[:2])


def pal_rgba16(pal):
    """RGBA16 TLUT bytes; opaque entries keep a=1, the transparent slot is 0x0000."""
    return texfmt.encode(pal.reshape(1, -1, 4), texfmt.RGBA, texfmt.B16)


def mio0_compress(data):
    """crunch64 (fast); it panics on some inputs (uniform 16 KB), then our own encoder."""
    try:
        return crunch64.mio0.compress(data)
    except BaseException:
        from cleanroom.codec import mio0
        return mio0.compress(data)


COMP_SIZE = {}   # symbol -> our MIO0 size (the game reads exactly TEX_COMPRESSED_SIZE bytes)
SIZES = {"raw": 0, "mio0": 0, "n_mio0": 0}


def write_tex(out, e, data):
    p = os.path.join(out, e["file"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if e.get("mio0"):
        comp = mio0_compress(bytes(data))
        COMP_SIZE[e["sym"]] = len(comp)
        SIZES["mio0"] += len(comp)
        SIZES["n_mio0"] += 1
        txt = fmt_inc(comp, "u8")
    else:
        SIZES["raw"] += len(data)
        txt = fmt_inc(data, e["ctype"])
    with open(p, "w", newline="\n") as f:
        f.write(txt)


def gen_textures(out):
    spec = json.load(open(os.path.join(SPEC, "textures.json"), encoding="utf-8"))
    from . import theme
    for sym, e in spec.items():
        e["sym"] = sym
        if e.get("pic"):
            # hooks colour/cut their drawings from a grid and a 2-bit mask: both now come from OUR kit
            e["grid"] = theme.grid(sym, e, e.get("gridn", 4))
            if e.get("alpha"):
                e["alpha2"] = theme.alpha2(sym, e)
    imgs, n = {}, 0
    from cleanroom.gfx import png as pngw
    for sym, e in spec.items():
        if e["fmt"] != "TLUT" and e.get("pic"):
            imgs[sym] = picture(sym, e)
            gp = os.path.join(out, "gate", "images", e.get("yaml", "misc"), sym + ".png")
            os.makedirs(os.path.dirname(gp), exist_ok=True)
            pngw.write(gp, imgs[sym])          # originality gate input (clean side)
    done_ci = set()
    for sym, e in spec.items():
        if e["fmt"] != "TLUT":
            continue
        users = [u for u in e["users"] if u in imgs]
        if not users:
            write_tex(out, e, pal_rgba16(np.zeros((e["colors"], 4), np.uint8)))
            continue
        size = e["colors"]
        cap = 16 if any(spec[u]["fmt"] == "CI4" for u in users) else 256
        k = min(size, cap)
        has_a = any((imgs[u][..., 3] < 128).any() for u in users)
        pal = build_palette([imgs[u] for u in users], k, has_a)
        full = np.zeros((size, 4), np.uint8)
        full[:k] = pal
        full[k:] = pal[-1]
        write_tex(out, e, pal_rgba16(full))
        for u in users:
            if u in done_ci:
                continue
            t = spec[u]
            f, s = FMTS[t["fmt"]]
            idx = index_image(imgs[u], pal, has_a)
            buf = np.zeros(idx.shape + (4,), np.uint8)
            buf[..., 0] = idx
            write_tex(out, t, texfmt.encode(buf, f, s))
            done_ci.add(u)
            n += 1
    for sym, e in spec.items():
        if e["fmt"] == "TLUT" or sym in done_ci or "grid" not in e:
            continue
        f, s = FMTS[e["fmt"]]
        img = imgs[sym]
        if f == texfmt.CI:
            buf = np.zeros_like(img)
            g = np.clip(np.round(img[..., 0].astype(np.float32)), 0, 15 if s == texfmt.B4 else 255)
            buf[..., 0] = g.astype(np.uint8)
            img = buf
        write_tex(out, e, texfmt.encode(img, f, s))
        n += 1
    print(f"textures written: {n} (+{sum(1 for e in spec.values() if e['fmt'] == 'TLUT')} TLUTs), "
          f"drawn by hooks: {HOOKED[0]}; raw {SIZES['raw']} B, mio0 {SIZES['n_mio0']} files {SIZES['mio0']} B "
          f"(retail mio0 {sum(e.get('mio0_bytes', 0) for e in spec.values())} B)")
    return imgs


# decomp C that hard-codes a retail MIO0 size instead of TEX_COMPRESSED_SIZE()
SRC_SIZE_PATCHES = [
    ("src/overlays/ovl_i6/credits.c", "aCopyrightTex, 0x439)", "aCopyrightTex, TEX_COMPRESSED_SIZE(aCopyrightTex))"),
]


def patch_comp_sizes(out):
    """Torch headers define _<tex>_COMPRESSED_SIZE with the retail MIO0 size: write ours."""
    n = 0
    hdir = os.path.join(out, "include", "assets", VER)
    for f in os.listdir(hdir):
        p = os.path.join(hdir, f)
        s = open(p).read()

        def rep(m):
            nonlocal n
            if m.group(1) in COMP_SIZE:
                n += 1
                return f"#define _{m.group(1)}_COMPRESSED_SIZE 0x{COMP_SIZE[m.group(1)]:x}"
            return m.group(0)

        s2 = re.sub(r"#define _(\w+)_COMPRESSED_SIZE 0x[0-9a-fA-F]+", rep, s)
        if s2 != s:
            open(p, "w", newline="\n").write(s2)
    for f, old, new in SRC_SIZE_PATCHES:
        p = os.path.join(out, f)
        s = open(p).read()
        if old in s:
            open(p, "w", newline="\n").write(s.replace(old, new))
    print(f"compressed sizes patched: {n} header defines (of {len(COMP_SIZE)} MIO0 textures)")
    # every object may use them
    shutil.rmtree(os.path.join(out, "build", VER, "src", "overlays"), ignore_errors=True)
    shutil.rmtree(os.path.join(out, "build", VER, "src", "game"), ignore_errors=True)


def gen_masks(out):
    """64x64 1-bit masks (ending fireworks): our own shapes from the mask spec (coarse occupancy)."""
    from . import masks
    p = os.path.join(out, "src", "assets", VER, "common_assets_compressed", "common_assets_compressed.c")
    s = open(p).read()
    n = [0]

    def rep(m):
        vals = re.findall(r"\d+", m.group(3))
        if len(vals) != 512:
            return m.group(0)
        n[0] += 1
        data = masks.mask_bytes(m.group(2))
        body = "\n" + "\n".join("    " + ", ".join(str(v) for v in data[i:i + 16]) + "," for i in range(0, 512, 16)) + "\n"
        return m.group(1) + body + m.group(4)

    s = MASK_RX.sub(rep, s)
    open(p, "w", newline="\n").write(s)
    print(f"1-bit masks regenerated: {n[0]}")


def machine_meshes(dirty, out):
    """New machine / object meshes (machgeo.py). The search takes minutes, so results are cached per machgeo.py
    version in D:/n64work/fzerox/cache/machgeo/<hash>/ (our generated C, not retail)."""
    import hashlib
    from . import machgeo
    h = hashlib.sha1(open(machgeo.__file__, "rb").read()).hexdigest()[:12]
    cache = os.path.join(os.environ.get("FZX_CACHE", "D:/n64work/fzerox/cache"), "machgeo", h)
    rel = [os.path.join("src", "assets", VER, f, f + ".c") for f in machgeo.FILES]
    if all(os.path.exists(os.path.join(cache, r)) for r in rel):
        for r in rel:
            shutil.copy2(os.path.join(cache, r), os.path.join(out, r))
        print(f"machine meshes: {len(rel)} files from cache {h}")
        return
    s = machgeo.write(dirty, out)
    for r in rel:
        os.makedirs(os.path.dirname(os.path.join(cache, r)), exist_ok=True)
        shutil.copy2(os.path.join(out, r), os.path.join(cache, r))
    print(f"machine meshes: regenerated ({s if not isinstance(s, dict) else {k: v for k, v in list(s.items())[:4]}})")


def main(argv):
    pristine, dirty, out = argv[:3]
    if not os.path.exists(out):
        shutil.copytree(pristine, out, ignore=shutil.ignore_patterns(".git", "cmake-build-release", "build"))
    copy_code(dirty, out)
    from . import boot
    boot.apply(out)
    from . import trackgen
    courses = trackgen.write(out)
    from . import sfxseq
    sfxseq.apply(out)
    from . import staffghost
    staffghost.write(out, courses)
    from . import textsrc
    textsrc.apply(pristine, out)
    machine_meshes(dirty, out)
    from . import tree_patches
    tree_patches.apply(out)
    try:
        from . import drawn
        HOOKS[:] = drawn.HOOKS
    except ImportError:
        pass
    WRAP.update(wrap_modes(out))
    print(f"wrap modes: {sum(1 for v in WRAP.values() if v[0] or v[1])} textures tile (periodic grid upsampling)")
    print(f"hooks: {[h.__module__.split('.')[-1] + '.' + h.__name__ for h in HOOKS]}")
    gen_textures(out)
    patch_comp_sizes(out)
    gen_masks(out)
    # make has no dependency on the #included .inc.c files: drop the asset objects so they rebuild
    shutil.rmtree(os.path.join(out, "build", VER, "src", "assets"), ignore_errors=True)
    from . import code_art
    code_art.apply(out)


if __name__ == "__main__":
    main(sys.argv[1:])
