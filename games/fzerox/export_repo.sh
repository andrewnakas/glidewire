#!/bin/sh
# Copy the F-Zero X clean-room sources into a standalone repo dir. Usage: export_repo.sh <repo dir>
R="$1"; mkdir -p "$R/games" "$R/ports/emu" "$R/tools/idowin"
copy_tree() { (cd "$1" && find . -type f ! -path "*/__pycache__/*" ! -name "*.pyc") | while read f; do mkdir -p "$2/$(dirname "$f")"; cp "$1/$f" "$2/$f"; done; }
copy_tree cleanroom "$R/cleanroom"
copy_tree games/fzerox "$R/games/fzerox"
rm -rf "$R/games/fzerox/sf64_ref"
cp games/__init__.py "$R/games/"
cp ports/emu/make_site.py ports/emu/shot.py ports/emu/touch.js ports/wasm/serve.py "$R/ports/emu/" && cp -r ports/emu/vendor "$R/ports/emu/"
cp tools/idowin/*.py tools/idowin/*.h "$R/tools/idowin/" 2>/dev/null
cp games/fzerox/README.md "$R/README.md"
cp STATUS.md "$R/STATUS.md"
printf '__pycache__/\n*.pyc\n*.z64\n*.n64\n*.v64\nbaserom.*\n' > "$R/.gitignore"
echo "exported: $(find "$R" -type f ! -path "*/.git/*" | wc -l) files -> $R"
