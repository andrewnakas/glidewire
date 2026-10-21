"""Originality gate: published music, text, geometry, images and code must be
clearly different from retail. Runs beside the taint scan; the report is
committed as ORIGINALITY.md and must show 0 failing before publishing.

    python -m cleanroom.originality <dirty> <clean> <spec> [--report ORIGINALITY.md] [--manifest F]

Manifest: games/<g>/originality.json (looked up as --manifest, else
<spec>/originality.json, else <spec>/../originality.json). Schema:

    {
      "game": "fzerox",
      "thresholds": {"music": {"dtw_sim_max": 0.6}, ...},     # optional overrides of config.CONFIG
      "music":     [{"retail": "assets/seq/*.m64", "clean": "assets/seq/*.m64", "format": "m64"}],
      "music_pcm": [{"retail": "audio/bgm/*.wav", "clean": "audio/bgm/*.wav"}],
      "text":      [{"retail": "text/us/*.json", "clean": "text/us/*.json"}],
      "names":     "names.json",
      "geometry":  [{"retail": "levels/*/area.obj", "clean": "levels/*/area.obj", "anchors": "anchors.json",
                     "thresholds": {"voxel": 0.5}}],                 # optional per-entry geometry overrides
      "tilemap":   [{"retail": "levels/*.json", "clean": "levels/*.json"}],   # 2D tile levels {"grid": rows, "anchor_cols": [...]}
      "images":    [{"retail": "textures/**/*.png", "clean": "textures/**/*.png"}],
      "code":      [{"retail": "baserom.z64", "clean": "build/game.z64", "start": "0x1000", "end": "0xF0000",
                     "clean_start": "0x1000", "clean_end": "0xF0000", "compiled": ["build/src/**/*.bin"]}]
    }

Paths: "retail" is relative to <dirty>, "clean" and "compiled" to <clean>,
"names" and "anchors" to the manifest's folder. Patterns are globs ("**"
recursive). music and text compare every clean file against every retail
file (a copy may move between slots); music_pcm, geometry and images pair
files by the path below the pattern's first wildcard. A pattern that
matches nothing is a failure (missing input), never a silent pass.
Formats: music by extension (.mid/.midi, .m64, else libultra cseq) or
"format"; text files are JSON (list of strings, {id: string}, or list of
{"text": ...}) or plain text, one string per line; meshes .obj/.npz/.json
(verts + tris); anchors JSON is a list of [x, y, z] or {"anchors": [...]}
with optional per-mesh lists under {"by_mesh": {"<key>": [...]}}.
"""
