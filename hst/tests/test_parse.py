"""Standalone (no-pytest) tests for the HST chapter parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # hst/ for build_hst_codes
import fixtures as F  # noqa: E402
import build_hst_codes as B  # noqa: E402


def by_code(rows):
    return {r["pct_code"]: r for r in rows}


def test_chapter_row():
    rows = B.parse_chapter(F.CH01_HTML)
    d = by_code(rows)
    assert d["01"]["level"] == "chapter"
    assert d["01"]["description_raw"] == "Live animals"
    assert d["01"]["parent_code"] is None


def test_heading_rows():
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert set(["0101", "0102", "0103", "0104", "0105", "0106"]) <= set(d)
    assert d["0101"]["level"] == "heading"
    assert d["0101"]["parent_code"] == "01"
    assert d["0101"]["description_raw"] == "Live horses, asses, mules and hinnies"


def test_real_hs6_with_bare_code():
    # 0102.29 prints a bare 6-digit row, name "Other", not synthetic.
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert d["0102.29"]["level"] == "subheading"
    assert d["0102.29"]["parent_code"] == "0102"
    assert d["0102.29"]["description_raw"] == "Other"
    assert d["0102.29"]["is_synthetic"] is False


def test_synthesized_hs6_from_eight_digit():
    # 0102.21 appears only as 0102.21.00 -> synthesized HS6.
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert d["0102.21"]["level"] == "subheading"
    assert d["0102.21"]["is_synthetic"] is True
    assert d["0102.21"]["description_raw"] == "Purebred breeding animals"


def test_grouping_label_in_full_desc():
    # "Cattle" colSpan label must be spliced into 0102.21 / 0102.29 full desc,
    # and must NOT exist as its own row.
    rows = B.parse_chapter(F.CH01_HTML)
    d = by_code(rows)
    assert "Cattle" in d["0102.21"]["description_full"]
    assert d["0102.21"]["description_full"].startswith(
        "Live animals > Live bovine animals"
    )
    assert all("Cattle" != r["pct_code"] for r in rows)
    assert "Cattle" not in d["0102.29"]["description_full"]


def test_no_national_rows():
    rows = B.parse_chapter(F.CH01_HTML)
    assert all(r["level"] in ("chapter", "heading", "subheading") for r in rows)
    # no 8/10-digit codes leaked
    assert all(len(r["pct_code"].replace(".", "")) <= 6 for r in rows)


def test_single_line_heading_creates_heading_row():
    # 0205000000 in the h2 tag — should still produce heading row "0205".
    rows = B.parse_chapter(F.CH02_HTML)
    d = by_code(rows)
    assert "0205" in d, "heading 0205 must exist; single-line h2 with 10-digit code was dropped"
    assert d["0205"]["level"] == "heading"
    assert d["0205"]["parent_code"] == "02"


def test_single_line_heading_hs6_parented_correctly():
    # 0205.00 must be a subheading under 0205, not under 0204.
    rows = B.parse_chapter(F.CH02_HTML)
    d = by_code(rows)
    assert "0205.00" in d, "0205.00 subheading must exist"
    assert d["0205.00"]["level"] == "subheading"
    assert d["0205.00"]["parent_code"] == "0205", (
        f"0205.00 parent must be 0205, got {d['0205.00']['parent_code']!r}"
    )


def test_single_line_heading_no_wrong_ancestry():
    # 0205.00 must NOT inherit 0204 "sheep or goats" ancestry.
    rows = B.parse_chapter(F.CH02_HTML)
    d = by_code(rows)
    assert "0205.00" in d
    full = d["0205.00"]["description_full"]
    assert "sheep" not in full.lower(), (
        f"0205.00 description_full must not contain 'sheep'; got: {full!r}"
    )
    assert "Meat of horses" in full or "horses" in full.lower(), (
        f"0205.00 must contain horse-meat text; got: {full!r}"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
