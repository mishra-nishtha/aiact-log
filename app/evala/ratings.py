"""Human-rating statistics for the API. Reads results/rating_key.json plus results/rating_A.csv (and rating_B.csv),
copies of rating_sheet.csv filled by raters. Same columns and ordinal Krippendorff alpha as scripts/score_ratings.py."""
import csv
import json
import os
from statistics import mean

from .metrics import krippendorff_alpha_ordinal

COLS = ("correctness_1to5", "completeness_1to5")

def _load(path, key):
    rows = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["item_id"] in rows: raise ValueError(f"{os.path.basename(path)}: duplicate item_id {r['item_id']}")
            rows[r["item_id"]] = r
    bad = sorted(set(rows) - set(key))
    if bad: raise ValueError(f"{os.path.basename(path)}: unknown item ids {bad[:5]} (sheet from a different run?)")
    for i, r in rows.items():
        for c in COLS:
            v = (r.get(c) or "").strip()
            if v and (not v.isdigit() or not 1 <= int(v) <= 5): raise ValueError(f"{i} {c}: {v!r} is not an integer 1-5")
        if (r.get("class_TG_ID_US") or "").strip().upper() not in ("", "TG", "ID", "US"):
            raise ValueError(f"{i}: class_TG_ID_US must be TG, ID or US")
    return rows

def report(res_dir):
    kp = os.path.join(res_dir, "rating_key.json")
    paths = [p for p in (os.path.join(res_dir, n) for n in ("rating_A.csv", "rating_B.csv")) if os.path.exists(p)]
    if not os.path.exists(kp) or not paths: return {"missing": True}
    with open(kp, encoding="utf-8") as f: key = json.load(f)
    R = [_load(p, key) for p in paths]
    val = lambda r, i, c: (r[i].get(c) or "").strip()
    out = {"raters": len(R), "metrics": {}}
    for c in COLS:
        per = {}
        for cond in ("B2", "C3", "C4"):
            v = [mean(int(val(r, i, c)) for r in R if val(r, i, c)) for i in key
                 if key[i]["cond"] == cond and any(i in r and val(r, i, c) for r in R)]
            per[cond] = {"mean": round(mean(v), 2) if v else None, "n": len(v)}
        m = {"by_condition": per}
        if len(R) >= 2:
            ids = [i for i in key if all(i in r and val(r, i, c) for r in R[:2])]
            if ids:
                a = [int(val(R[0], i, c)) for i in ids]; b = [int(val(R[1], i, c)) for i in ids]
                al = krippendorff_alpha_ordinal([[x, y] for x, y in zip(a, b)])
                m["agreement"] = {"n": len(ids), "exact": round(mean(x == y for x, y in zip(a, b)), 3),
                                  "within_1": round(mean(abs(x - y) <= 1 for x, y in zip(a, b)), 3),
                                  "krippendorff_alpha_ordinal": None if al != al else round(al, 3), "paper_target": 0.70}
        out["metrics"][c] = m
    if len(R) >= 2:
        ids = [i for i in key if all(i in r and val(r, i, "class_TG_ID_US") for r in R[:2])]
        if ids:
            out["class_agreement_raw"] = {"n": len(ids), "value": round(mean(
                val(R[0], i, "class_TG_ID_US").upper() == val(R[1], i, "class_TG_ID_US").upper() for i in ids), 3)}
    return out