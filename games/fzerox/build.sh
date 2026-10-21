#!/bin/sh
# Build the fzerox decomp ROM natively on Windows (Git Bash).
#   games/fzerox/build.sh <tree> [make args...]
# Needs: ~/bin shims (tools/setup_winbin.sh), libdragon mips64-elf binutils, native IDO passes
# (7.1: D:/n64work/fzerox/idowin71/bin via tools/idowin/build71.bat; 5.3: the PW64 idowin/bin)
# driven by tools/idowin/ido_cc.py, the decomp venv (D:/n64work/fzerox/venv).
set -e
# C: is full on this machine: keep every temp file (IDO passes, asm-processor, Python) on D:
export TMP=D:/n64work/fzerox/tmp TEMP=D:/n64work/fzerox/tmp TMPDIR=D:/n64work/fzerox/tmp
HERE="$(cd "$(dirname "$0")/../.." && pwd -W 2>/dev/null || pwd)"
T="$1"; shift
export PATH="$HOME/bin:$HOME/.local/mips64/bin:$PATH" PYTHONUTF8=1
I71="${IDO71_BIN:-D:/n64work/fzerox/idowin71/bin}"
I53="${IDO53_BIN:-C:/Users/andre/n64work/idowin/bin}"
PY=C:/Users/andre/AppData/Local/Programs/Python/Python312/python.exe
cd "$T"
make ${TARGET_RULE:-all} -j${JOBS:-2} RUN_CC_CHECK=0 COMPARE=0 N_THREADS=${JOBS:-2} PYTHON=D:/n64work/fzerox/venv/Scripts/python.exe \
  IDO="$PY $HERE/tools/idowin/ido_cc.py --bin=$I71" IDO53="$PY $HERE/tools/idowin/ido_cc.py --bin=$I53" "$@"
