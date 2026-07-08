"""Plain-text HS6 code search over the `codes` table — NO embeddings.

Grep `description_full` for keywords, restricted to level='subheading' (HS6 leaf).
Instant (5,756 rows, no model load), for grounding a labeler's HS6 pick in REAL
corpus codes. Default output is TERSE: `pct_code  description_raw` (the short own
label) to keep tool output small. Use --full ONLY to fetch the full ancestor
chain for the ONE code you finally pick (to store as true_desc).

Usage:
    python search_codes.py leather footwear         # AND-match all keywords (default)
    python search_codes.py --or bag purse handbag   # OR-match any keyword
    python search_codes.py --prefix 6403            # all subheadings under a heading
    python search_codes.py --full 6403.99           # full description_full for one code

Default/OR/prefix print:  <pct_code>  <description_raw>   (short own label)
--full prints:            <description_full>              (full chain, for storage)
"""
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent / "hst_corpus.db"


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: search_codes.py <kw...> | --or <kw...> | --prefix NNNN | --full CODE")
        sys.exit(1)

    con = sqlite3.connect(DB)

    if args[0] == "--full":
        # full ancestor chain for one exact code (to store as true_desc)
        code = args[1]
        row = con.execute(
            "SELECT description_full FROM codes WHERE pct_code = ?", (code,)
        ).fetchone()
        con.close()
        print(row[0] if row else f"!! no such code: {code}")
        return

    if args[0] == "--prefix":
        rows = con.execute(
            "SELECT pct_code, description_raw FROM codes "
            "WHERE level='subheading' AND pct_code LIKE ? ORDER BY pct_code",
            (args[1] + "%",),
        ).fetchall()
    else:
        disj = args[0] == "--or"
        kws = args[1:] if disj else args
        if not kws:
            print("no keywords")
            sys.exit(1)
        joiner = " OR " if disj else " AND "
        clause = joiner.join(["description_full LIKE ?"] * len(kws))
        params = [f"%{k}%" for k in kws]
        rows = con.execute(
            f"SELECT pct_code, description_raw FROM codes "
            f"WHERE level='subheading' AND ({clause}) ORDER BY pct_code",
            params,
        ).fetchall()

    con.close()
    for code, raw in rows:
        print(f"{code}  {raw or '(no own label)'}")
    print(f"[{len(rows)} subheadings]")


if __name__ == "__main__":
    main()
