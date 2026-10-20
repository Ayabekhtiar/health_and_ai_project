"""Hand-checked karyotypes from the training file -> expected IPSS-R category.

Run: .venv/bin/python -m pytest -q tests   (or: .venv/bin/python tests/test_cytogenetics.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mds_tool.cytogenetics import parse  # noqa: E402

CASES = [
    ("46,xy[20]", "Good"),                                   # normal
    ("46,xx", "Good"),
    ("45,x,-y[20]", "Very good"),                            # -Y single
    ("minus_y", "Very good"),
    ("46,xx,del(11)(q14q23)[20]", "Very good"),              # del(11q) single
    ("46,xx,del(5)(q13q33)[12]/46,xx[8]", "Good"),           # isolated del(5q)
    ("46,xy,del(20)(q12)[2]/46,xy[18]", "Good"),             # del(20q)
    ("46,xx,del(12)(p12p13)", "Good"),                       # del(12p)
    ("46,xy,del(5)(q13q33),+8[14]/46,xy[11]", "Good"),       # double incl. del(5q)
    ("47,xy,+8", "Intermediate"),                            # +8
    ("46,xx,del(7)(q22q32)[10]", "Intermediate"),            # del(7q) single
    ("46,xy,i(17)(q10)[15]", "Intermediate"),                # i(17q)
    ("46,xy,t(12;15)(q13;q22)[14]/46,xy[4]", "Intermediate"),  # other single
    ("47,xx,del(20)(q11),+8[10]", "Intermediate"),           # other double
    ("45,xx,-7[8]/46,xx[15]", "Poor"),                       # -7
    ("46,xy,inv(3)(q21q26)[20]", "Poor"),                    # inv(3)
    ("46,xy,t(3;3)(q21;q26)[8]/46,xy[12]", "Poor"),          # t(3q)
    ("46,xx,del(5)(q13q33),-7[10]/46,xx[14]", "Poor"),       # double incl. -7
    ("47,xy,+8,del(20)(q11),-y", "Poor"),                    # 3 abnormalities
    ("45,xx,del(5)(q13q33),-7,-17,+mar[19]/46,xx[1]", "Very poor"),  # >3
    (">3abnormalities", "Very poor"),
    ("46,xy,9ph", "Good"),                                   # heteromorphism = normal
    ("46,xx,asxl-1undrunx-1mutation[13]", "Good"),           # molecular note, not karyotype
    ("46,xy[20]/einzelaberrationen:46,idem,del(11)(q23)[1]/46,idem,del(20)(q11)[1]", "Good"),
    ("46,xy,inv(9)(p11;q13)[1]/47,idem,+8[19]einzelaberrationen:12p-,13q-", "Intermediate"),
]


def test_categories():
    for raw, expected in CASES:
        got = parse(raw).category
        assert got == expected, f"{raw!r}: expected {expected}, got {got}"


def test_missing_and_fish_only():
    assert parse(None).category is None
    assert parse(float("nan")).category is None
    assert parse("nucish5q31(egr1x2)[100]").category is None


def test_flags():
    k = parse("45,xy,del(5)(q14q34),del(17)(p12),-7[10]")
    assert k.flags["del5q"] and k.flags["del17_17p"] and k.flags["minus_7"]
    assert parse("46,xy,i(17)(q10)").flags["del17_17p"]


if __name__ == "__main__":
    test_categories(); test_missing_and_fish_only(); test_flags()
    print(f"OK: {len(CASES)} karyotypes + edge cases")
