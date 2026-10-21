"""Text that lives in the decomp's C source (names, menu words, credits, messages) -> our own wording.

    python -m games.fzerox.textsrc scan <pristine> <gate dir>    # DIRTY: string table for the gate (not committed)
                                                                   #        + spec/strings.json (ids + byte budgets only)
    python -m games.fzerox.textsrc check                           # every scanned id has a rewrite within budget

apply(out) (from generate.py) replaces each literal in the clean tree by id with text/strings.json.
An id is "<source file>#<n>": the n-th string literal of that file that is game text (string literals
inside #include lines, PRINTF / osSyncPrintf debug calls, and four-letter file tags are skipped).
Rewrites are our own words (the name map in names.json); the budget is 1.5x the retail byte length + 4, at least 18
(on-screen boxes are sized by the code), never below 2.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "spec", "strings.json")
TEXT = os.path.join(HERE, "text", "strings.json")
DIRS = ["src/game", "src/overlays", "src/sys"]
SKIP_FILES = ("sys_fault.c", "leo_fault.c", "leo_fault_dd.c")       # crash screen: register names / formats
LIT = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
DEBUG = re.compile(r"PRINTF|osSyncPrintf|Fault_|ASSERT|__FILE__|#include|#pragma|INCLUDE_ASM|INCLUDE_RODATA|GLOBAL_ASM")
TAG = re.compile(r"^[A-Z0-9_]{4}$")


def _files(root):
    for d in DIRS:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            for f in sorted(fn):
                if f.endswith(".c") and f not in SKIP_FILES:
                    yield os.path.relpath(os.path.join(dp, f), root).replace("\\", "/")


def is_text(s):
    """Game text: has a letter, is not a format-only or tag string."""
    plain = re.sub(r"%[-0-9.]*[a-zA-Z]", "", s)
    return bool(re.search(r"[A-Za-z　-￿]", plain)) and not TAG.match(s)


def literals(text):
    """[(start, end, body)] of game-text literals in a C file, in order."""
    out = []
    for ln_start, line in _lines(text):
        if DEBUG.search(line):
            continue
        for m in LIT.finditer(line):
            if is_text(m.group(1)):
                out.append((ln_start + m.start(1), ln_start + m.end(1), m.group(1)))
    return out


def _lines(text):
    pos = 0
    for line in text.split("\n"):
        yield pos, line
        pos += len(line) + 1


def scan(root, gate):
    ids, table = {}, {}
    for rel in _files(root):
        t = open(os.path.join(root, rel), encoding="utf8", errors="replace").read()
        for k, (a, b, s) in enumerate(literals(t)):
            i = f"{rel}#{k}"
            ids[i] = max(18, len(s.encode("utf8")) * 3 // 2 + 4)
            table[i] = s
    os.makedirs(gate, exist_ok=True)
    json.dump(table, open(os.path.join(gate, "text_retail.json"), "w", encoding="utf8"), ensure_ascii=False, indent=0)
    json.dump({"note": "string literal ids in the decomp source and their byte budgets; wording is ours "
                       "(text/strings.json)", "budget": ids}, open(SPEC, "w"), indent=0)
    print(f"textsrc scan: {len(ids)} literals in {len(set(i.split('#')[0] for i in ids))} files -> {SPEC}, "
          f"{gate}/text_retail.json")


def _c_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"') if "\\" not in s else s


def apply(pristine, out):
    """Rewrite from the pristine decomp file each time (ids are literal positions in the decomp source)."""
    budget = json.load(open(SPEC))["budget"]
    new = json.load(open(TEXT, encoding="utf8"))
    files = {}
    for i in budget:
        files.setdefault(i.split("#")[0], []).append(i)
    n = 0
    clean = []
    for rel, ids in files.items():
        p = os.path.join(out, rel)
        t = open(os.path.join(pristine, rel), encoding="utf8", errors="replace").read()
        lits = literals(t)
        assert len(lits) == len(ids), f"{rel}: {len(lits)} literals, spec has {len(ids)}"
        for k in range(len(lits) - 1, -1, -1):
            a, b, s = lits[k]
            i = f"{rel}#{k}"
            r = new.get(i)
            if r is None:
                raise SystemExit(f"textsrc: no rewrite for {i}")
            t = t[:a] + r + t[b:]
            clean.append(r)
            n += 1
        open(p, "w", encoding="utf8", newline="\n").write(t)
    os.makedirs(os.path.join(out, "gate"), exist_ok=True)
    json.dump(clean, open(os.path.join(out, "gate", "text_clean_src.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=0)
    print(f"textsrc: {n} source strings rewritten in {len(files)} files")


def check():
    budget = json.load(open(SPEC))["budget"]
    new = json.load(open(TEXT, encoding="utf8"))
    missing = [i for i in budget if i not in new]
    over = [i for i in budget if i in new and len(new[i].encode("utf8")) > budget[i]]
    print(f"textsrc check: {len(budget)} ids, {len(missing)} missing, {len(over)} over budget "
          f"{(missing + over)[:6]}")
    return not missing and not over


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "check":
        sys.exit(0 if check() else 1)
