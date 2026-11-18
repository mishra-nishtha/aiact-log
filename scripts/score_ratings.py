"""Human ratings -> means per condition + inter-rater agreement.
python scripts/score_ratings.py results/rating_A.csv results/rating_B.csv   (copies of rating_sheet.csv filled by two raters)"""
import csv, json, sys
from statistics import mean
key = json.load(open("results/rating_key.json"))
def load(p): return {r["item_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
R = [load(p) for p in sys.argv[1:]]
cols = ["correctness_1to5", "completeness_1to5", "clarity_1to5"]
def qwk(a, b, k=5):
    n = len(a); O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b): O[x - 1][y - 1] += 1
    ra = [sum(r) for r in O]; cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum((i - j) ** 2 * O[i][j] for i in range(k) for j in range(k)); den = sum((i - j) ** 2 * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")
for c in cols:
    print(f"\n{c}")
    for cond in ("B2", "C3", "C4"):
        v = [mean(int(r[i][c]) for r in R if r[i][c]) for i in key if key[i]["cond"] == cond and all(r[i][c] for r in R)]
        print(f"  {cond}: mean {mean(v):.2f} (n={len(v)})" if v else f"  {cond}: no ratings")
    if len(R) >= 2:
        ids = [i for i in key if all(r[i][c] for r in R[:2])]; a = [int(R[0][i][c]) for i in ids]; b = [int(R[1][i][c]) for i in ids]
        if ids: print(f"  agreement raters 1-2: exact {mean(x == y for x, y in zip(a, b)):.2f}, within-1 {mean(abs(x - y) <= 1 for x, y in zip(a, b)):.2f}, quadratic-weighted kappa {qwk(a, b):.2f}")
