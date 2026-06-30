"""Validate the built hst_corpus.db: structure, FKs, HS6 ceiling, spot checks."""
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "hst_corpus.db"


def main() -> int:
    con = sqlite3.connect(DB)
    q = con.execute
    fails = []

    # no national / no >6-digit codes
    leaked = q(
        "select count(*) from codes where level not in "
        "('chapter','heading','subheading')"
    ).fetchone()[0]
    if leaked:
        fails.append(f"{leaked} non-HS6 level rows")

    # every heading has a valid chapter parent
    bad_h = q(
        "select count(*) from codes h where h.level='heading' and h.parent_code "
        "not in (select pct_code from codes where level='chapter')"
    ).fetchone()[0]
    if bad_h:
        fails.append(f"{bad_h} headings with bad parent")

    # every subheading has a valid heading parent
    bad_s = q(
        "select count(*) from codes s where s.level='subheading' and "
        "s.parent_code not in (select pct_code from codes where level='heading')"
    ).fetchone()[0]
    if bad_s:
        fails.append(f"{bad_s} subheadings with bad parent")

    # cd_percent all NULL; fiscal_year all 2026
    if q("select count(*) from codes where cd_percent is not null").fetchone()[0]:
        fails.append("non-null cd_percent present")
    if q("select count(*) from codes where fiscal_year<>'2026'").fetchone()[0]:
        fails.append("fiscal_year != 2026 present")

    # chapter-01 spot checks
    def raw(code):
        r = q("select description_raw from codes where pct_code=?", (code,)).fetchone()
        return r[0] if r else None

    def full(code):
        r = q("select description_full from codes where pct_code=?", (code,)).fetchone()
        return r[0] if r else None

    if raw("0102.29") != "Other":
        fails.append("0102.29 raw != 'Other'")
    if raw("0102.21") != "Purebred breeding animals":
        fails.append("0102.21 raw mismatch")
    if "Cattle" not in (full("0102.21") or ""):
        fails.append("Cattle label missing from 0102.21 full desc")

    counts = dict(
        q("select level, count(*) from codes group by level").fetchall()
    )
    print("level counts:", counts)
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("OK - all corpus validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
