"""Build-tool patches so the fzerox decomp builds natively on Windows (Git Bash + make).

    python -m games.fzerox.tree_patches <tree>           # before extract/build
    python -m games.fzerox.tree_patches <tree> --splat   # after splat (backslash paths)

Each patch: (file, old, new). Idempotent (skips when `new` is already present).
Toolchain: native IDO 7.1 + 5.3 passes (tools/idowin/ido_cc.py --bin=...), libdragon mips64-elf
binutils, Torch built with MSVC, crunch64 (python) for MIO0.
"""
import os
import sys

PATCHES = [
    ("Makefile",
     "ifeq ($(OS),Windows_NT)\n$(error Native Windows is currently unsupported for building this repository, use WSL instead c:)\n",
     "ifeq ($(OS),Windows_NT)\n    DETECTED_OS := windows\n"),
    # Torch builds with MSVC into cmake-build-release/Release/
    ("Makefile", "TORCH           := $(TOOLS)/Torch/cmake-build-release/torch\n",
     "TORCH           := $(TOOLS)/Torch/cmake-build-release/Release/torch.exe\n"),
    # the compiler paths come from the make command line (IDO=..., IDO53=...)
    ("Makefile", "  CC       := $(TOOLS)/ido-recomp/$(DETECTED_OS)/7.1/cc\n", "  CC       := $(IDO)\n"),
    ("Makefile", "IDO53           := $(TOOLS)/ido-recomp/$(DETECTED_OS)/5.3/cc\n", "IDO53           ?= $(TOOLS)/ido-recomp/$(DETECTED_OS)/5.3/cc\n"),
    ("Makefile", "IDO             := $(TOOLS)/ido-recomp/$(DETECTED_OS)/7.1/cc\n", "IDO             ?= $(TOOLS)/ido-recomp/$(DETECTED_OS)/7.1/cc\n"),
    # MIO0 decompression with crunch64 instead of the C tool
    ("tools/decompressMio0Segments.py", "        run([mio0, \"-d\", mio0in, mio0out])",
     "        import crunch64\n"
     "        open(mio0out, 'wb').write(crunch64.mio0.decompress(open(mio0in, 'rb').read()))"),
    # IDO's cfe reads backslashes in paths (#line markers, -I) as string escapes: use forward slashes
    ("tools/asm-processor/build.py", "asmproc_flags += opt_flags + [str(in_file)]",
     "asmproc_flags += opt_flags + [in_file.as_posix()]"),
    ("tools/asm-processor/build.py", '+ ["-I", str(in_dir), "-o", str(out_file), str(preprocessed_path)]',
     '+ ["-I", in_dir.as_posix(), "-o", out_file.as_posix(), preprocessed_path.as_posix()]'),
    ("tools/asm-processor/build.py", "            str(out_file),\n", "            out_file.as_posix(),\n"),
    ("tools/asm-processor/build.py", "    try:\n        subprocess.check_call(compile_cmdline)",
     "    import shutil as _sh\n"
     "    compile_cmdline[0] = _sh.which(compile_cmdline[0]) or compile_cmdline[0]\n"
     "    try:\n        subprocess.check_call(compile_cmdline)"),
]


def fix_splat_paths(tree):
    """splat on Windows writes backslash paths into .incbin lines and the linker script."""
    n = 0
    for d in ("asm", "linker_scripts"):
        for root, _, files in os.walk(os.path.join(tree, d)):
            for f in files:
                if not f.endswith((".s", ".ld", ".d")):
                    continue
                p = os.path.join(root, f)
                s = open(p, newline="").read()
                if "\\" in s:
                    open(p, "w", newline="").write(s.replace("\\", "/"))
                    n += 1
    print(f"splat paths: {n} files normalised")


def apply(tree):
    n = 0
    for f, old, new in PATCHES:
        p = os.path.join(tree, f)
        s = open(p, newline="").read()
        if new in s:
            continue
        assert old in s, f"patch target missing in {f}: {old[:60]!r}"
        open(p, "w", newline="").write(s.replace(old, new, 1))
        n += 1
    print(f"tree_patches: {n} applied, {len(PATCHES) - n} already present")


if __name__ == "__main__":
    if "--splat" in sys.argv:
        fix_splat_paths(sys.argv[1])
    else:
        apply(sys.argv[1])
