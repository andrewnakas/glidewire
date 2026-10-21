"""DEV ONLY (bisecting): copy dirty texture files for matching symbols into the clean tree.

    python -m games.fzerox.dev_hybrid <clean> <dirty> <symbol regex> [--mio0 only|none]
Writes <clean>/DEV_RETAIL_AUDIO (make_site refuses to publish). Re-run generate.py afterwards.
"""
import json
import os
import re
import shutil
import sys

from .generate import SPEC, VER


def main(argv):
    clean, dirty, rx = argv[:3]
    mode = argv[argv.index("--mio0") + 1] if "--mio0" in argv else "any"
    spec = json.load(open(os.path.join(SPEC, "textures.json"), encoding="utf-8"))
    n = 0
    for sym, e in spec.items():
        if not re.search(rx, sym) and not re.search(rx, e["yaml"]):
            continue
        if (mode == "only" and not e.get("mio0")) or (mode == "none" and e.get("mio0")):
            continue
        shutil.copy2(os.path.join(dirty, e["file"]), os.path.join(clean, e["file"]))
        n += 1
    open(os.path.join(clean, "DEV_RETAIL_AUDIO"), "w").write("dev hybrid\n")
    shutil.rmtree(os.path.join(clean, "build", VER, "src", "assets"), ignore_errors=True)
    print(f"dev hybrid: {n} dirty texture files copied")


if __name__ == "__main__":
    main(sys.argv[1:])
