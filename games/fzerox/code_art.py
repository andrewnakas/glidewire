"""CLEAN ROOM: pixel data that the decomp keeps inside its C source (not extracted by Torch).

  src/sys/sys_fault.c       the crash-screen 5x7 font bitmap (redrawn from our own glyph table, faultfont.py)
"""
import os
import re

from . import faultfont


def _replace_array(src, name, values, per=12, width=4):
    m = re.search(r"(\b%s\[[^\]]*\]\s*=\s*\{)(.*?)(\n\};)" % re.escape(name), src, re.S)
    assert m, name
    body = "\n" + "\n".join("    " + " ".join(f"0x{v:0{width}X}," for v in values[i:i + per])
                            for i in range(0, len(values), per))
    return src[:m.start(2)] + body + src[m.end(2):]


def apply(out):
    p = os.path.join(out, "src/sys/sys_fault.c")
    s = open(p).read()
    s = _replace_array(s, "sFaultCharPixelFlags", faultfont.pixel_flags(), 8, 8)
    open(p, "w", newline="\n").write(s)
    print("code art: sys_fault font redrawn")
