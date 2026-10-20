"""Parse ISCN karyotype strings into IPSS-R cytogenetic risk categories.

The CYTOGENETICS column is free text written by 23 laboratories. Most entries
follow ISCN (e.g. "46,xy,del(20)(q12)[2]/46,xy[18]"), a few are shorthand
("complex", "del5q", "tris8"). The parser is deliberately conservative: every
string it cannot read is reported as missing rather than guessed.

IPSS-R cytogenetic categories (Schanz et al., JCO 2012):
  Very good    : -Y, del(11q)                          (single)
  Good         : normal, del(5q), del(12p), del(20q)   (single); double incl. del(5q)
  Intermediate : del(7q), +8, +19, i(17q), any other single or double
  Poor         : -7, inv(3)/t(3q)/del(3q); double incl. -7/del(7q); 3 abnormalities
  Very poor    : more than 3 abnormalities
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CATEGORIES = ["Very good", "Good", "Intermediate", "Poor", "Very poor"]
CYTO_CODE = {name: i for i, name in enumerate(CATEGORIES)}

# Free-text shorthands seen in the file, mapped to an equivalent ISCN token list.
SHORTHAND = {
    "complex": ["complex3"],
    "complex[18]": ["complex3"],
    "complexkaryotype": ["complex3"],
    "complexkaryptype": ["complex3"],
    ">3abnormalities": ["complex4"],
    "dely": ["-y"],
    "minus_y": ["-y"],
    "tris8": ["+8"],
    "plus8": ["+8"],
    "+8(fish)": ["+8"],
    "del7": ["del(7)(q)"],
    "del20": ["del(20)(q)"],
    "del(20q)": ["del(20)(q)"],
    "del(5q)": ["del(5)(q)"],
    "del5q": ["del(5)(q)"],
    "del5q-": ["del(5)(q)"],
    "5q-isolata": ["del(5)(q)"],
    "del11q": ["del(11)(q)"],
}

_CELLS = re.compile(r"\[[^\]]*\]")
_PLOIDY = re.compile(r"^(<[0-9n]+>|\d{2,4}([~-]\d{2,3})?(<\d?n>)?(xy|xx)?)$")
_SEX = re.compile(r"^[xy]{1,4}c?$")
_SKIP = {"idem", "sl", "sdl", "sdl1", "sdl2", "inc", "cp", "?", ""}


@dataclass
class Karyotype:
    raw: str | None
    parsed: bool = False
    abnormalities: list[str] = field(default_factory=list)
    n_abn: int = 0
    category: str | None = None   # one of CATEGORIES, or None if unreadable / missing
    flags: dict = field(default_factory=dict)


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [t.strip() for t in out]


def _is_noise(t: str) -> bool:
    """Tokens that are not acquired clonal abnormalities."""
    return bool(
        "mutation" in t or "klinefelter" in t
        or re.match(r"^\d+[pq]h[+-]?$", t)              # heteromorphism, e.g. 9ph, 1qh+
        or re.match(r"^inv\(9\)\(p1\d?;?q1\d?\)$", t)   # pericentric inv(9): normal variant
        or re.match(r"^(\d+[pq]\d|cen\d)", t)           # bare FISH probe names
    )


def _tokens(raw: str) -> list[str] | None:
    s = raw.strip().lower().replace(" ", "")
    if s in SHORTHAND:
        return SHORTHAND[s]
    s = s.replace("minus_y", "-y").replace("_", "")
    # German "Einzelaberrationen" = single-cell (non-clonal) findings: not counted
    s = re.split(r"/?einzelaberrationen", s)[0] or "46,xx"
    if s.startswith("nucish"):
        return None  # interphase FISH only: not a karyotype
    if "keinezelleninteilung" in s:  # German: "no metaphases"; FISH shows 5q-
        return ["del(5)(q)"] if "5q-" in s else None
    if ";46," in s or s.startswith("5q-;"):
        # semi-colon separated free text, e.g. "5q-;t(2;11)[8]40%;t(2;11)[9];46,xx[3]"
        s = s.replace("5q-", "del(5)(q)")
        parts = re.split(r";(?![^()]*\))", s)
        s = "/".join("46,xx," + p if not p.startswith("46") else p for p in parts)
    toks: list[str] = []
    for clone in s.split("/"):
        clone = _CELLS.sub("", clone)
        clone = re.sub(r"\d+%", "", clone)
        for t in _split_top_level(clone):
            if not t or t in _SKIP or _PLOIDY.match(t) or _SEX.match(t) or _is_noise(t):
                continue
            if t.endswith("c") and re.match(r"^[+-]\w+c$", t):
                continue  # constitutional change, not part of the disease
            t = re.sub(r"x\d+$", "", t)  # +8x2 -> +8
            toks.append(t)
    return toks


def _count(tokens: list[str]) -> int:
    n = 0
    for t in tokens:
        m = re.match(r"^\+(\d+)(~\d+)?mar$", t)
        if m:
            n += int(m.group(1))
        elif t == "complex3":
            n += 3
        elif t == "complex4":
            n += 4
        else:
            n += 1
    return n


def _arm_hits(tokens: list[str], chrom: str, arm: str, kinds: tuple[str, ...]) -> bool:
    """True if a structural abnormality of `kinds` hits chromosome `chrom` arm `arm`."""
    for t in tokens:
        m = re.match(r"^(\w+)\(([^)]*)\)\(([^)]*)\)", t)
        if not m:
            m2 = re.match(r"^(\w+)\((\w+?)([pq])\)$", t)  # shorthand like del(5q)
            if m2 and m2.group(1) in kinds and m2.group(2) == chrom and m2.group(3) == arm:
                return True
            continue
        kind, chroms, bands = m.groups()
        if kind not in kinds:
            continue
        cs, bs = chroms.split(";"), bands.split(";")
        for i, c in enumerate(cs):
            b = bs[i] if i < len(bs) else bs[-1]
            if c == chrom and b.lstrip("?").startswith(arm):
                return True
    return False


def parse(raw) -> Karyotype:
    if raw is None or (isinstance(raw, float) and raw != raw) or str(raw).strip() == "":
        return Karyotype(raw=None)
    toks = _tokens(str(raw))
    if toks is None:
        return Karyotype(raw=str(raw))
    # unique abnormalities across clones (stemline repeated in subclones)
    uniq = list(dict.fromkeys(toks))
    k = Karyotype(raw=str(raw), parsed=True, abnormalities=uniq, n_abn=_count(uniq))

    f = k.flags
    f["minus_y"] = "-y" in uniq
    f["minus_7"] = "-7" in uniq
    f["del7q"] = _arm_hits(uniq, "7", "q", ("del",))
    f["del5q"] = _arm_hits(uniq, "5", "q", ("del",))
    f["del11q"] = _arm_hits(uniq, "11", "q", ("del",))
    f["del12p"] = _arm_hits(uniq, "12", "p", ("del",))
    f["del20q"] = _arm_hits(uniq, "20", "q", ("del",))
    f["plus_8"] = "+8" in uniq
    f["plus_19"] = "+19" in uniq
    f["i17q"] = any(t.startswith("i(17)(q") or t == "i(17q)" for t in uniq)
    f["abn3q"] = (_arm_hits(uniq, "3", "q", ("inv", "t", "del", "ins"))
                  or any(t.startswith("inv(3)") for t in uniq))
    f["del17_17p"] = ("-17" in uniq or f["i17q"]
                      or _arm_hits(uniq, "17", "p", ("del", "add", "der", "t")))
    f["del7_7q"] = f["minus_7"] or f["del7q"]
    f["complex"] = k.n_abn >= 3
    k.category = _ipssr_category(k)
    return k


def _ipssr_category(k: Karyotype) -> str:
    f, n = k.flags, k.n_abn
    if n == 0:
        return "Good"
    if n > 3:
        return "Very poor"
    if n == 3:
        return "Poor"
    if f["minus_7"] or f["abn3q"]:
        return "Poor"
    if n == 2:
        if f["del7_7q"]:
            return "Poor"
        if f["del5q"]:
            return "Good"
        return "Intermediate"
    # single abnormality
    if f["minus_y"] or f["del11q"]:
        return "Very good"
    if f["del5q"] or f["del12p"] or f["del20q"]:
        return "Good"
    return "Intermediate"
