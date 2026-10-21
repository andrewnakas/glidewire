# Glidewire — an original anti-gravity racer built on the fzerox decomp

Glidewire is compiled from the [fzerox decompilation](https://github.com/inspectredc/fzerox) (F-Zero X US 1.0
code) and runs in the browser on [N64Wasm](https://github.com/nbarkhina/N64Wasm). The program is the decomp's
compiled code; **everything you see and hear is our own** (originality rebuild, October 2026):

| Part | What we made | Kept from retail (structure / code anchors only) |
|---|---|---|
| Courses (26 slots) | `trackgen.py`: our loops (turtle walks, figure eights), heights, widths, segment shapes (walls, tunnels, pipes, drums, bowls, air gaps), banking, every feature lane; checked with the game's own spline maths; CPU guidance tables | per slot: venue, skybox, music slot, control-point count |
| Machines, podium, course objects | `machgeo.py`: new shapes from each mesh's own connectivity | vertex counts, triangle lists (in the decomp's draw code), coarse boxes |
| Cast | 30 new pilots and craft (`cast.py`, `names.json`) | which picture slot a pilot fills |
| Textures (1529) | our texture kit (`theme.py`) and drawing hooks (labels, fonts, HUD, portraits, signs, logo, skylines) | size, format, segment, transparency flag, coarse cover fraction |
| Text | every source string, texture label and announcer line in our words (`textsrc.py`, `text/strings.json`, `text_labels.json`, `voice_lines.json`) | string slots |
| Music | 18 tracks from our seeded composer (`music.py`) | slot length and rate |
| Sound effects | designed from recipes (`sfx.py`) | length, rate, loop flag, loudness, median pitch, tonal/noisy |
| Voices | Piper TTS placeholders of our own lines | — |
| Boot | our IPL3 stub (CC0) and CRUCODE1 microcode placeholders (`boot.py`, `cleanroom.ucode`) | 33 IPL3-slot words the game code reads as constants (only the bits it reads) |

Gates: `ORIGINALITY.md` (music, text, geometry, images, retail-code windows) and `TAINT.md` (asset byte runs,
plus the whole ROM against the retail data the rebuild replaced) — both must show 0 failing.

The ROM carries no retail RSP microcode, so it needs an emulator core patched for CRUCODE1 tags
(n64cleanrecomp `ports/emu/ucode-tags`); the site ships such an N64Wasm build. It does not run on real
hardware (unsigned IPL3, placeholder microcode).

Controls: arrow keys = stick · `D` = A (accelerate) · `S` = B (brake) · `A` / `E` = Z / R · `Enter` = Start ·
`I` = C-up (view) · gamepads work too.

## Build (Windows, Git Bash)

Needs Python 3 (numpy, scipy, Pillow, crunch64, piper-tts for voices), GNU make, libdragon mips64-elf binutils,
native IDO 7.1 / 5.3 passes (`tools/idowin`), Torch (MSVC) and the decomp's Python requirements.

```sh
# dirty room, once: needs your own ROM, never published
git clone https://github.com/inspectredc/fzerox dirty       # + submodules, core.autocrlf=false
python -m games.fzerox.tree_patches dirty
cp baserom.us.rev0.z64 dirty/ && TARGET_RULE=extract sh games/fzerox/build.sh dirty
python -m games.fzerox.tree_patches dirty --splat && TARGET_RULE=assets sh games/fzerox/build.sh dirty
python -m games.fzerox.extract_spec dirty ; python -m games.fzerox.audio spec dirty
python -m games.fzerox.courses spec dirty/src/assets/us/rev0/course_data/course_data.c pristine/include games/fzerox/spec/courses.json dirty
python -m games.fzerox.textsrc scan pristine <gate dir>

# clean room
games/fzerox/build_clean.sh --audio <site dir>
python -m games.fzerox.gate_inputs dirty clean <gate dir>
python -m cleanroom.originality <gate dir> <gate dir> games/fzerox/spec --manifest games/fzerox/originality.json --report ORIGINALITY.md
python -m games.fzerox.taint_report dirty clean > TAINT.md
python tools/race_check.py clean <dev tree> <out>            # dev: CPU races every course (IS-Viewer log)
```
