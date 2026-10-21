#!/bin/sh
# Clean pipeline: generate assets -> build ROM -> web site.
#   games/fzerox/build_clean.sh [--audio] [site dir]
# --audio also regenerates the samples (slow: ~25 min on 4 workers).
set -e
# C: is full on this machine: keep every temp file (IDO passes, asm-processor, Python) on D:
export TMP=D:/n64work/fzerox/tmp TEMP=D:/n64work/fzerox/tmp TMPDIR=D:/n64work/fzerox/tmp
set -o pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
W=D:/n64work/fzerox
SITE=$W/site
AUDIO=0
for a in "$@"; do case $a in --audio) AUDIO=1;; *) SITE=$a;; esac; done
cd "$R"
export PYTHONUTF8=1
python -m games.fzerox.generate $W/pristine $W/dirty $W/clean 2>&1 | grep -v "Warning\|return fun"
[ $AUDIO = 1 ] && python -m games.fzerox.audio gen $W/clean
[ -f $W/clean/bin/us/rev0/audio_table.bin ] || { echo "no clean audio yet: run with --audio"; exit 1; }
[ -f $W/clean/DEV_RETAIL_AUDIO ] && { echo "clean tree is a dev hybrid: regenerate audio"; exit 1; }
# included data isn't a make dependency: drop objects whose inputs we regenerate
rm -rf $W/clean/build/us/rev0/src/assets $W/clean/build/us/rev0/bin $W/clean/build/us/rev0/src/sys/sys_fault.o \
  $W/clean/build/us/rev0/src/rsp $W/clean/build/us/rev0/src/leo/lib $W/clean/build/us/rev0/asm/us/rev0/header.o
sh games/fzerox/build.sh $W/clean > $W/build_clean.log 2>&1 || { grep -m5 "rror" $W/build_clean.log; exit 1; }
ROM=$W/clean/build/us/rev0/fzerox.us.rev0.z64
ls -la $ROM | awk '{print "rom", $5, "bytes"}'
python -m games.fzerox.segsize $W/dirty $W/clean | tail -1
python ports/emu/make_site.py "$SITE" $ROM
