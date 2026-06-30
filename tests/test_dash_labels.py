#!/usr/bin/env python3
"""Standalone (no-pytest) tests for dash-label path reconstruction in build_codes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for build_codes
sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests dir for fixtures
import build_codes
import fixtures as F


def build_rows_from_text(raw: str) -> dict:
    stream = build_codes.parse_with_labels(raw)
    return build_codes.build_db(stream, chapter_titles={})


def _full(rows, code):
    return rows[code]["description_full"]


def test_4202_siblings_distinct():
    rows = build_rows_from_text(F.FIX_4202)
    a = _full(rows, "4202.2100")
    b = _full(rows, "4202.3100")
    c = _full(rows, "4202.9100")
    assert a != b != c and a != c, "4202.21/31/91 description_full must differ"
    assert "Handbags" in a, a
    assert "carried in the pocket" in b, b
    # 4202.91 sits under the bare "- Other:" group label
    assert b != c


def test_2501_rock_salt_in_chain():
    rows = build_rows_from_text(F.FIX_2501)
    assert "Rock salt" in _full(rows, "2501.0021"), _full(rows, "2501.0021")
    assert "Pink rock salt" in _full(rows, "2501.0021")
    # sibling NOT under rock salt must not gain it
    assert "Rock salt" not in _full(rows, "2501.0030"), _full(rows, "2501.0030")


def test_8544_both_labels_in_chain():
    rows = build_rows_from_text(F.FIX_8544)
    full = _full(rows, "8544.3011")
    assert "Ignition wiring sets" in full, full
    assert "Of a kind used in vehicles of chapter 87" in full, full


def test_4402_unchanged():
    rows = build_rows_from_text(F.FIX_4402)
    assert "Of bamboo" in _full(rows, "4402.1000")
    assert "Of shell or nut" in _full(rows, "4402.2000")
    assert "Other" in _full(rows, "4402.9000")
    # the three must differ
    s = {_full(rows, c) for c in ("4402.1000", "4402.2000", "4402.9000")}
    assert len(s) == 3, s


def test_0205_heading_is_leaf():
    rows = build_rows_from_text(F.FIX_0205)
    assert rows["0205.0000"]["level"] == "heading"
    assert "Meat of horses" in _full(rows, "0205.0000")


def test_0101_mixed_depths():
    rows = build_rows_from_text(F.FIX_0101)
    # 0101.21 under "- Horses:" group; 0101.30 "- Asses" is its own subheading
    assert "Horses" in _full(rows, "0101.2100"), _full(rows, "0101.2100")
    assert "Pure-bred breeding animals" in _full(rows, "0101.2100")
    assert "Asses" in _full(rows, "0101.3000"), _full(rows, "0101.3000")
    assert "Horses" not in _full(rows, "0101.3000"), _full(rows, "0101.3000")


def test_subheading_label_paths():
    stream = build_codes.parse_with_labels(F.FIX_4202)
    sl = build_codes.subheading_labels(stream)
    # 4202.21 is grouped under the depth-1 "Handbags..." label; its own label is
    # "With outer surface of leather..."
    assert "4202.21" in sl
    assert any("Handbags" in p for p in sl["4202.21"]["path"]), sl["4202.21"]
    assert "leather" in sl["4202.21"]["raw"].lower(), sl["4202.21"]
    # 4202.11 has TWO leaves (.1120/.1190) at depth-3 under "Trunks..." + a depth-2
    # split; the shared prefix is the depth-1 "Trunks..." label.
    assert any("Trunks" in p for p in sl["4202.11"]["path"]), sl["4202.11"]

def test_national_label_path_8544():
    stream = build_codes.parse_with_labels(F.FIX_8544)
    sl = build_codes.subheading_labels(stream)
    sub_path = sl["8544.30"]["path"]
    # find the 8544.3011 leaf object
    leaf = next(o for k, o in stream if k == "leaf" and o.code == "8544.3011")
    np = build_codes.national_label_path(leaf, sub_path + [sl["8544.30"]["raw"]])
    assert any("chapter 87" in x for x in np), np


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
