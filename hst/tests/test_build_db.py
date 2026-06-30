"""Standalone test: parse_chapter -> build_db round-trip on chapter 01."""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixtures as F  # noqa: E402
import build_hst_codes as B  # noqa: E402


def test_build_db_roundtrip():
    rows = B.parse_chapter(F.CH01_HTML)
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        B.build_db(rows, db)
        con = sqlite3.connect(db)
        # schema columns match PCT exactly
        cols = [r[1] for r in con.execute("PRAGMA table_info(codes)")]
        assert cols == [
            "pct_code", "level", "parent_code", "description_raw",
            "description_full", "cd_percent", "fiscal_year", "is_synthetic",
        ]
        n_chap = con.execute(
            "select count(*) from codes where level='chapter'"
        ).fetchone()[0]
        n_head = con.execute(
            "select count(*) from codes where level='heading'"
        ).fetchone()[0]
        assert n_chap == 1 and n_head == 6
        # cd_percent NULL, fiscal_year set
        fy, cd = con.execute(
            "select fiscal_year, cd_percent from codes where pct_code='0102.29'"
        ).fetchone()
        assert fy == "2026" and cd is None
        # referential sanity: every subheading parent is a heading
        bad = con.execute(
            "select count(*) from codes s where s.level='subheading' and "
            "s.parent_code not in (select pct_code from codes where level='heading')"
        ).fetchone()[0]
        assert bad == 0


if __name__ == "__main__":
    test_build_db_roundtrip()
    print("PASS test_build_db_roundtrip\n\n1 passed")
