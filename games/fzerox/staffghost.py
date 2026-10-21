"""CLEAN ROOM: our own crew (staff) ghosts. Each one is a time-trial run of the game's own CPU driver on OUR
course, recorded by the game's own ghost recorder in the dev race-check ROM (tools/race_check.py --ghosts)
and stored here as games/fzerox/ghosts/<slot>.json (our data; no retail replay is used).

    python -m games.fzerox.staffghost import <race_log.txt> [...]   # log -> ghosts/<slot>.json
    write(out, courses)   generate.py: src/assets/us/rev0/staff_ghost_records/staff_ghost_records.c

A slot without a recording gets a short idle ghost (it never matches a course, so it is never raced) with a
generous crew time, so unlocks stay reachable.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GHOSTS = os.path.join(HERE, "ghosts")
VER = "us/rev0"
NAMES = ["MuteCity1", "Silence1", "SandOcean1", "DevilsForest1", "BigBlue1", "PortTown1", "SectorAlpha",
         "RedCanyon1", "DevilsForest2", "MuteCity2", "BigBlue2", "WhiteLand1", "FireField", "Silence2",
         "SectorBeta", "RedCanyon2", "WhiteLand2", "MuteCity3", "RainbowRoad", "DevilsForest3", "SpacePlant",
         "SandOcean2", "PortTown2", "BigHand"]
CHARACTERS = ["CAPTAIN_FALCON", "DR_STEWART", "PICO", "SAMURAI_GOROH", "JODY_SUMMER", "MIGHTY_GAZELLE",
              "MR_EAD", "BABA", "OCTOMAN", "GOMAR_AND_SHIOH", "KATE_ALEN", "ROGER_BUSTER", "JAMES_MCCLOUD",
              "LEON", "ANTONIO_GUSTER", "BLACK_SHADOW", "MICHAEL_CHAIN", "JACK_LEVIN", "SUPER_ARROW",
              "MRS_ARROW", "JOHN_TANAKA", "BEASTMAN", "ZODA", "DR_CLASH", "SILVER_NEELSEN", "BIO_REX", "DRAQ",
              "BILLY", "THE_SKULL", "BLOOD_FALCON"]


def import_log(paths):
    os.makedirs(GHOSTS, exist_ok=True)
    n = 0
    for p in paths:
        cur = None
        for line in open(p, encoding="utf8"):
            m = re.match(r"GHOST (\d+) size=(\d+) end=(\d+) time=(\d+) l=(-?\d+),(-?\d+),(-?\d+)", line)
            if m:
                if cur:
                    _save(cur)
                    n += 1
                v = [int(x) for x in m.groups()]
                cur = dict(slot=v[0], size=v[1], end=v[2], time=v[3], laps=v[4:7], data="")
            elif line.startswith("GD ") and cur is not None:
                cur["data"] += line[3:].strip()
        if cur:
            _save(cur)
            n += 1
    print(f"staffghost: {n} recordings imported -> {GHOSTS}")


def _save(g):
    if len(g["data"]) // 2 < g["size"] or g["time"] <= 0:
        print(f"  slot {g['slot']}: incomplete recording skipped")
        return
    g["data"] = g["data"][:2 * g["size"]]
    json.dump(g, open(os.path.join(GHOSTS, f"{g['slot']:02d}.json"), "w"))


def _s8(b):
    return b - 256 if b >= 128 else b


def write(out, courses):
    lines = ['#include "macros.h"', '#include ASSET_HEADER(staff_ghost_records.h)', '#include "fzx_game.h"',
             '#include "fzx_machine.h"', '#include "fzx_save.h"', ""]
    real = 0
    for slot, nm in enumerate(NAMES):
        p = os.path.join(GHOSTS, f"{slot:02d}.json")
        if os.path.exists(p):
            g = json.load(open(p))
            data = bytes.fromhex(g["data"])
            enc = ((courses[slot]["checksum"] << 5) | slot) & 0xFFFFFFFF
            time, laps, end = g["time"], g["laps"], g["end"]
            real += 1
        else:
            data = bytes(6)
            enc, time, laps, end = 0, 180000, [60000, 60000, 60000], 6
        words = [int.from_bytes((data + bytes(4))[i:i + 4], "big", signed=True) for i in range(0, len(data) - len(data) % 4 or 4, 4)]
        rchk = sum(words) & 0xFFFFFFFF
        ch = CHARACTERS[(slot * 7 + 3) % 30]
        rgb = [(slot * 37) % 200 + 40, (slot * 91) % 200 + 40, (slot * 53) % 200 + 40]
        lines += [f"GhostRecord a{nm}StaffGhostRecord = {{",
                  "    0x0000, GHOST_STAFF,",
                  f"    0x{rchk:08X}, 0x{enc:08X}, {time},",
                  '    2, { 0 }, "",',
                  f"    {{ {ch}, CUSTOM_MACHINE_DEFAULT, FRONT_0, REAR_0, WING_NONE, LOGO_SHIELD, MACHINE_NUMBER({slot + 1}),",
                  f"      DECAL_STRIPE, {rgb[0]}, {rgb[1]}, {rgb[2]}, 250, 250, 250, 20, 20, 30, 40, 200, 230 }}",
                  "};", "",
                  f"GhostReplayInfo a{nm}StaffGhostReplayInfo = {{",
                  f"    0x0000, 0, {{ {laps[0]}, {laps[1]}, {laps[2]} }}, {end}, {len(data)}, 0, 0",
                  "};", "",
                  f"s8 a{nm}StaffGhostData[] = {{"]
        for i in range(0, len(data), 24):
            lines.append("    " + ", ".join(str(_s8(b)) for b in data[i:i + 24]) + ",")
        lines += ["};", ""]
    p = os.path.join(out, "src", "assets", VER, "staff_ghost_records", "staff_ghost_records.c")
    open(p, "w", newline="\n").write("\n".join(lines))
    print(f"staff ghosts: {real} recorded on our courses, {24 - real} idle placeholders")


if __name__ == "__main__":
    if sys.argv[1] == "import":
        import_log(sys.argv[2:])
