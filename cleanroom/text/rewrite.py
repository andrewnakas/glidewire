"""Budgeted string rewrite: every dialogue/UI string gets new wording that
fits the slot's byte budget and keeps its control codes in order.

No writing happens here. The flow is:
  1. dirty room exports strings.json:  [{"id", "text", "budget"?, "context"?}] or {id: text}
     (budget = bytes the slot holds; default = the encoded retail length)
  2. `--brief brief.json` writes what the authoring pass (LLM + owner review)
     gets: id, budget, control codes, context, and the name-mapped line
     (`--no-source` withholds the line itself)
  3. the authoring pass returns rewrites.json: {id: "new text"}
  4. this tool validates (missing, unencodable, over budget, control codes
     changed, >= 5 shared words with any retail string, retail name kept)
     and emits the game's table through a pluggable encoder.

    python -m cleanroom.text.rewrite <strings.json> <rewrites.json> [--names names.json]
        [--encoder ascii|utf8|charmap:<map.json>] [--out table.bin] [--c out.c --symbol name]
        [--brief brief.json [--no-source]]     (rewrites.json may be - with --brief)

Charmap JSON: {"chars": {"A": "0a", ...}, "controls": {"{END}": "ff", ...},
"terminator": "ff"}. Any encoder also accepts raw hex control tokens {0A0B}.
"""
import json
import re
import sys
from dataclasses import dataclass

from cleanroom.originality import text as otext
from cleanroom.originality.config import CONFIG

HEX_TOKEN = re.compile(r"\{((?:[0-9A-Fa-f]{2})+)\}")


@dataclass
class Entry:
    id: str
    text: str
    budget: int = None
    context: str = ""


class Encoder:
    def __init__(self, base="ascii", chars=None, controls=None, terminator=b"", control_re=None):
        self.base, self.chars, self.controls = base, chars or {}, controls or {}
        self.terminator = terminator
        self.control_re = re.compile(control_re or CONFIG["text"]["control_re"])

    @classmethod
    def named(cls, spec, terminator=b"\x00"):
        if spec.startswith("charmap:"):
            d = json.load(open(spec[8:], encoding="utf8"))
            return cls("charmap", {k: bytes.fromhex(v) for k, v in d.get("chars", {}).items()},
                       {k: bytes.fromhex(v) for k, v in d.get("controls", {}).items()},
                       bytes.fromhex(d.get("terminator", "")), d.get("control_re"))
        return cls(spec, terminator=terminator)

    def tokens(self, s):
        pos = 0
        for m in self.control_re.finditer(s):
            if m.start() > pos:
                yield False, s[pos:m.start()]
            yield True, m.group(0)
            pos = m.end()
        if pos < len(s):
            yield False, s[pos:]

    def encode(self, s):
        out = bytearray()
        for is_ctl, tok in self.tokens(s):
            if is_ctl:
                if tok in self.controls:
                    out += self.controls[tok]
                elif HEX_TOKEN.fullmatch(tok):
                    out += bytes.fromhex(HEX_TOKEN.fullmatch(tok).group(1))
                else:
                    out += self._plain(tok)
            else:
                out += self._plain(tok)
        return bytes(out + self.terminator)

    def _plain(self, s):
        if self.base == "charmap":
            try:
                return b"".join(self.chars[c] for c in s)
            except KeyError as e:
                raise ValueError(f"character {e.args[0]!r} not in charmap") from None
        return s.encode("utf-8" if self.base == "utf8" else "ascii")


def control_codes(s, cfg=None):
    return re.findall((cfg or CONFIG["text"])["control_re"], s)


def apply_names(s, names):
    for k in sorted(names, key=len, reverse=True):
        s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])", names[k], s)
    return s


def load_strings(path):
    d = json.load(open(path, encoding="utf8"))
    if isinstance(d, dict):
        return [Entry(str(k), v) for k, v in d.items()]
    return [Entry(str(x["id"]), x["text"], x.get("budget"), x.get("context", "")) for x in d]


def budget(e, enc):
    return e.budget if e.budget is not None else len(enc.encode(e.text))


def brief(entries, names, enc, with_source=True, cfg=None):
    out = []
    for e in entries:
        b = {"id": e.id, "budget": budget(e, enc), "controls": control_codes(e.text, cfg), "context": e.context}
        if with_source:
            b["source_name_mapped"] = apply_names(e.text, names)
        out.append(b)
    return out


def validate(entries, rewrites, enc, names=None, cfg=None):
    """-> list of (id, kind, message); kinds: missing, encode, budget,
    controls, overlap, name."""
    cfg = cfg or CONFIG["text"]
    names = names or {}
    probs = []
    retail = [e.text for e in entries]
    for e in entries:
        if e.id not in rewrites:
            probs.append((e.id, "missing", "no rewrite"))
            continue
        new = rewrites[e.id]
        try:
            n = len(enc.encode(new))
            if n > budget(e, enc):
                probs.append((e.id, "budget", f"{n} B > {budget(e, enc)} B"))
        except ValueError as err:
            probs.append((e.id, "encode", str(err)))
        if control_codes(new, cfg) != control_codes(e.text, cfg):
            probs.append((e.id, "controls", f"{control_codes(new, cfg)} != {control_codes(e.text, cfg)}"))
        for _, w in otext.shared_runs([new], retail, cfg):
            probs.append((e.id, "overlap", f'shares "{w}"'))
        for _, k in otext.name_hits([new], names, cfg):
            probs.append((e.id, "name", f"keeps retail name {k}"))
    return probs


def emit_table(entries, rewrites, enc, pad=True):
    """Concatenated encoded strings (in entry order) + their offsets. With
    pad, each string is padded with terminator/zero bytes to its budget so
    fixed-slot tables keep their layout."""
    blob, offs = bytearray(), []
    for e in entries:
        b = enc.encode(rewrites[e.id])
        offs.append(len(blob))
        if pad:
            b = b.ljust(budget(e, enc), enc.terminator[:1] or b"\x00")
        blob += b
    return bytes(blob), offs


def emit_c(symbol, entries, rewrites, enc):
    lines = [f"/* generated by cleanroom.text.rewrite: {len(entries)} strings */"]
    for i, e in enumerate(entries):
        b = enc.encode(rewrites[e.id])
        body = ", ".join(f"0x{x:02X}" for x in b)
        lines.append(f"static const unsigned char {symbol}_{i}[] = {{ {body} }}; /* {e.id} */")
    lines.append(f"const unsigned char *{symbol}[] = {{ " + ", ".join(f"{symbol}_{i}" for i in range(len(entries))) + " };")
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    entries = load_strings(argv[0])
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    enc = Encoder.named(opt("--encoder", "ascii"))
    names = json.load(open(opt("--names"), encoding="utf8")) if opt("--names") else {}
    if opt("--brief"):
        json.dump(brief(entries, names, enc, "--no-source" not in argv), open(opt("--brief"), "w", encoding="utf8"),
                  indent=1, ensure_ascii=False)
        print(f"rewrite: brief for {len(entries)} strings -> {opt('--brief')}")
        if argv[1] == "-" or argv[1].startswith("--"):
            return 0
    rewrites = json.load(open(argv[1], encoding="utf8"))
    probs = validate(entries, rewrites, enc, names)
    kinds = {}
    for _, k, _ in probs:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"rewrite: {len(entries)} strings, {len(rewrites)} rewrites, {len(probs)} problems {kinds or ''}")
    for i, k, m in probs[:10]:
        print(f"  {k:8s} {i}: {m[:100]}")
    if probs:
        return 1
    if opt("--out"):
        blob, _ = emit_table(entries, rewrites, enc)
        open(opt("--out"), "wb").write(blob)
        print(f"  table -> {opt('--out')} ({len(blob)} B)")
    if opt("--c"):
        open(opt("--c"), "w").write(emit_c(opt("--symbol", "strings"), entries, rewrites, enc))
        print(f"  C -> {opt('--c')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
