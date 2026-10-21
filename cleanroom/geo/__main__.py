"""python -m cleanroom.geo demo <out.c> [--seed N] [--anchors anchors.json]

Builds a kit level (terrain flattened at the anchors, a few blocks and a
ramp, a banked loop track) and writes Vtx/Gfx + collision C; one-line summary.
"""
import sys

import numpy as np

from cleanroom.geo import anchors as A, f3dex, kit


def demo(seed=0, pts=None):
    rng = np.random.default_rng(seed)
    terrain = kit.heightfield(4000, 4000, 200, seed, 300, anchors=pts, origin=(-2000, 0, -2000), surface=1)
    blocks = [kit.box(p, p + rng.uniform(150, 500, 3), surface=2)
              for p in rng.uniform(-1800, 1500, (4, 3)) * [1, 0, 1] + [0, 300, 0]]
    ramp = kit.ramp((-200, 300, -200), 300, 600, 200, surface=3)
    ring = [(1500 * np.cos(a), 450 + 100 * np.sin(3 * a), 1500 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 9)]
    track = kit.spline_track(ring, 300, surface=4)
    return kit.merge(terrain, *blocks, ramp, track)


def main(argv):
    if len(argv) < 2 or argv[0] != "demo":
        print(__doc__)
        return 2
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    pts = A.points(A.load(argv[argv.index("--anchors") + 1])) if "--anchors" in argv else None
    m = demo(seed, pts)
    src = f3dex.emit_c("kit_level", m) + "\n" + f3dex.collision_c("kit_level_col", m)
    open(argv[1], "w").write(src)
    lo, hi = m.bounds()
    print(f"geo demo: {len(m.verts)} verts {len(m.tris)} tris, {len(f3dex.batches(m.tris))} vertex batches, "
          f"surfaces {sorted(set(m.surface.tolist()))}, bounds {lo.astype(int)}..{hi.astype(int)} -> {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
