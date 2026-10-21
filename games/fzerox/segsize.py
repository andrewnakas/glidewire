"""Compare MIO0-compressed segment sizes (clean vs dirty build): our data must not compress worse than
the retail data, the game decompresses into fixed buffers.

    python -m games.fzerox.segsize <dirty tree> <clean tree>
"""
import re
import sys

import crunch64

SEGS = ["course_track_gfx", "machine_models", "mute_city_textures", "port_town_textures", "big_blue_textures",
        "sand_ocean_textures", "devils_forest_textures", "white_land_textures", "sector_textures",
        "red_canyon_textures", "fire_field_textures", "silence_textures", "ending_venue_textures", "podium_gfx"]


def seg_sizes(tree):
    mp = open(f"{tree}/build/us/rev0/fzerox.us.rev0.map").read()
    rom = open(f"{tree}/build/us/rev0/fzerox_uncompressed.us.rev0.z64", "rb").read()
    out = {}
    for s in SEGS:
        a = int(re.search(r"(0x[0-9a-f]+)\s+%s_ROM_START = " % s, mp).group(1), 16)
        b = int(re.search(r"(0x[0-9a-f]+)\s+%s_ROM_END = " % s, mp).group(1), 16)
        out[s] = (b - a, len(crunch64.mio0.compress(rom[a:b])))
    return out


def main(dirty, clean):
    d, c = seg_sizes(dirty), seg_sizes(clean)
    bad = 0
    for s in SEGS:
        flag = "OVER" if c[s][1] > d[s][1] else ""
        bad += bool(flag)
        print(f"{s:24s} raw {d[s][0]:7d}/{c[s][0]:7d}  mio0 retail {d[s][1]:7d} clean {c[s][1]:7d} {flag}")
    print(f"segments over retail compressed size: {bad}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
