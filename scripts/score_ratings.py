"""Human ratings -> means per condition + inter-rater agreement (exact, within-1, QWK, Krippendorff alpha ordinal).
python scripts/score_ratings.py results/rating_A.csv results/rating_B.csv   (copies of rating_sheet.csv filled by two raters)"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statistics import mean

from app.evala.metrics import krippendorff_alpha_ordinal

with open("results/rating_key.json", encoding="utf-8") as f: key = json.load(f)
COLS = ["correctness_1to5", "completeness_1to5"]

def load(p):
    rows = {}
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["item_id"] in rows: sys.exit(f"{p}: duplicate item_id {r['item_id']}")
            rows[r["item_id"]] = r
    extra = set(rows) - set(key)
    if extra: sys.exit(f"{p}: unknown item ids {sorted(extra)[:5]} (sheet from a different run?)")
    for i, r in rows.items():
        for c in COLS:
            v = (r.get(c) or "").strip()
            if v and (not v.isdigit() or not 1 <= int(v) <= 5): sys.exit(f"{p} {i} {c}: {v!r} is not an integer 1-5")
        if (r.get("grounded_Y_N") or "").strip().upper() not in ("", "Y", "N"): sys.exit(f"{p} {i}: grounded_Y_N must be Y or N")
        if (r.get("class_TG_ID_US") or "").strip().upper() not in ("", "TG", "ID", "US"): sys.exit(f"{p} {i}: class_TG_ID_US must be TG, ID or US")
    return rows

R = [load(p) for p in sys.argv[1:]]
if not R: sys.exit("usage: score_ratings.py rating_A.csv [rating_B.csv]")
val = lambda r, i, c: (r[i].get(c) or "").strip()

def qwk(a, b, k=5):
    n = len(a); O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b): O[x - 1][y - 1] += 1
    ra = [sum(r) for r in O]; cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum((i - j) ** 2 * O[i][j] for i in range(k) for j in range(k)); den = sum((i - j) ** 2 * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")

for c in COLS:
    print(f"\n{c}")
    for cond in ("B2", "C3", "C4"):
        v = [mean(int(val(r, i, c)) for r in R) for i in key if key[i]["cond"] == cond and all(i in r and val(r, i, c) for r in R)]
        print(f"  {cond}: mean {mean(v):.2f} (n={len(v)})" if v else f"  {cond}: no ratings")
    if len(R) >= 2:
        ids = [i for i in key if all(i in r and val(r, i, c) for r in R[:2])]
        a = [int(val(R[0], i, c)) for i in ids]; b = [int(val(R[1], i, c)) for i in ids]
        if ids:
            al = krippendorff_alpha_ordinal([[x, y] for x, y in zip(a, b)])
            print(f"  agreement raters 1-2 (n={len(ids)}): exact {mean(x == y for x, y in zip(a, b)):.2f}, within-1 {mean(abs(x - y) <= 1 for x, y in zip(a, b)):.2f}, "
                  f"QWK {qwk(a, b):.2f}, Krippendorff alpha (ordinal) {al:.2f} (paper target >= 0.70)")
for name in ("grounded_Y_N", "class_TG_ID_US"):
    for n, r in enumerate(R, 1):
        v = [val(r, i, name).upper() for i in r if val(r, i, name)]
        if v and name == "grounded_Y_N": print(f"\nrater {n} {name}: {mean(x == 'Y' for x in v):.2f} Y (n={len(v)})")
    if len(R) >= 2:
        ids = [i for i in key if all(i in r and val(r, i, name) for r in R[:2])]
        if ids: print(f"{name} raw agreement raters 1-2: {mean(val(R[0], i, name).upper() == val(R[1], i, name).upper() for i in ids):.2f} (n={len(ids)})")