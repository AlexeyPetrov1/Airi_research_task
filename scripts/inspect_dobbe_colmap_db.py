"""Show strongest wide-baseline verified pairs in an RGB-only COLMAP DB."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

P = 2147483647


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--min-gap", type=int, default=20)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as con:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        match_rows = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        geometry_rows = con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        pairs = []
        for pair_id, n in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
            id1, id2 = divmod(pair_id, P)
            try:
                f1 = int(Path(names[id1]).stem)
                f2 = int(Path(names[id2]).stem)
            except (KeyError, ValueError):
                continue
            if abs(f1 - f2) >= args.min_gap:
                pairs.append((n, f1, f2, id1, id2))
    print("images:", len(names), "matches rows:", match_rows,
          "geometry rows:", geometry_rows,
          "expected exhaustive pairs:", len(names) * (len(names) - 1) // 2)
    print("wide verified pairs:", len(pairs))
    for n, f1, f2, id1, id2 in sorted(pairs, reverse=True)[:25]:
        print(f"{f1:04d} {f2:04d}: {n} inliers (IDs {id1}, {id2})")


if __name__ == "__main__":
    main()
