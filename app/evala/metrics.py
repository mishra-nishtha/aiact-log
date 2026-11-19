import itertools
import math
import random
import re
from statistics import mean, stdev

from .data import CORPUS, GOLD, GOLD_LINKS, UNITS
from .llm import retrieve_ids, toks

MATCH_T, GROUND_T = 0.6, 0.5
def keycov(text, keys):
    """Share of key-term groups present. Alternatives match as WORD PREFIXES (\\brecord matches 'recorded', not 'catalog')."""
    t = text.lower()
    return sum(any(re.search(r"\b" + re.escape(a), t) for a in k.split("|")) for k in keys) / len(keys)
def unit_ref_text(u): return " ".join([u["text"]] + [CORPUS[r][1] for r in u["refs"]])
def ground(text, u):
    w = [x for x in toks(text) if len(x) > 3]; ref = set(toks(unit_ref_text(u)))
    return sum(x in ref for x in w) / len(w) if w else 0

def score_derive(u, outs):
    gold = GOLD[u["id"]]
    rec = mean(any(keycov(o, g[2]) >= MATCH_T for o in outs) for g in gold) if outs else 0
    prec = mean(any(keycov(o, g[2]) >= MATCH_T for g in gold) for o in outs) if outs else 0
    uns = mean(ground(o, u) < GROUND_T for o in outs) if outs else 0
    return {"recall": rec, "precision": prec, "unsupported": uns, "n": len(outs)}

def norm_refs(items):
    out = set()
    for s in items or []:
        s = str(s).lower()
        for m in re.finditer(r"\b(?:article|art\.?)\s*(\d+)(?:\s*\((\d+)\))?", s): out.add(f"A{m.group(1)}" + (f"({m.group(2)})" if m.group(2) else ""))
        for m in re.finditer(r"\bannex\s+([ivx]+)\b", s): out.add("ANX" + m.group(1).upper())
    return out
def f1(p, g):
    if not p and not g: return 1.0
    tp = len(p & g); return 0 if not tp else 2 * tp / (len(p) + len(g))
def score_schema(u, ob):
    if not isinstance(ob, dict): return None
    st = str(ob.get("actor_status", "")).lower(); pred_stated = "stated" in st and "not" not in st and "un" not in st
    return {"xref_f1": f1(norm_refs(ob.get("cross_references")), set(u["cross_refs"])),
            "actor_ok": pred_stated == (u["actor_status"] == "stated"),
            "temporal_ok": bool(ob.get("temporal_constraints")) == u["temporal"],
            "exceptions_ok": bool(ob.get("exceptions")) == u["exceptions"],
            "conditions_ok": bool(ob.get("conditions")) == u["conditions"]}

def score_links(pred, unit_id):
    """pred: {gold_id: [artifact ids]} -> per-gold (tp,fp,fn) and a full-chain flag.
    Full chain = EVERY gold requirement of the unit has ALL of its reference links recovered."""
    per = {}
    for gid, *_ in GOLD[unit_id]:
        G, P = set(GOLD_LINKS[gid]), set(pred.get(gid, []))
        per[gid] = (len(G & P), len(P - G), len(G - P))
    return per, all(t[2] == 0 for t in per.values())

def prf(tuples):
    tp, fp, fn = (sum(t[i] for t in tuples) for i in range(3))
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0,
            "f2": 5 * p * r / (4 * p + r) if 4 * p + r else 0}
    
def retrieval_metrics(ks=(1, 3, 5)):
    """Retriever evaluated on its own. Query = unit text; relevant = the unit's reference corpus ids;
    the unit's parent provision is excluded. Units with no cross-references are skipped.
    Binary-relevance nDCG."""
    units = [u for u in UNITS if u["refs"]]
    rec = {k: [] for k in ks}; nd = {k: [] for k in ks}; rr = []
    for u in units:
        rel = set(u["refs"]); ranked = retrieve_ids(u["text"], max(ks), exclude={u["parent"]})
        first = next((i for i, c in enumerate(ranked, 1) if c in rel), None)
        rr.append(1 / first if first else 0)
        for k in ks:
            top = ranked[:k]
            rec[k].append(len(rel & set(top)) / len(rel))
            dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(top, 1) if c in rel)
            idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
            nd[k].append(dcg / idcg if idcg else 0)
    return {"n_units": len(units), "retriever": "tfidf-cosine", "mrr": round(mean(rr), 3) if rr else None,
            **{f"recall@{k}": round(mean(rec[k]), 3) if units else None for k in ks},
            **{f"ndcg@{k}": round(mean(nd[k]), 3) if units else None for k in ks},
            "note": "relevance = author-drafted refs; units without cross-references are not scored"}

def bootstrap_f1(tuples, n=1000, seed=7):
    rnd = random.Random(seed); xs = sorted(prf([rnd.choice(tuples) for _ in tuples])["f1"] for _ in range(n))
    return [xs[int(.025 * n)], xs[int(.975 * n)]]

def _ranks(a):
    o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def wilcoxon(x, y):
    d = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]; n = len(d)
    if n == 0: return {"p": 1.0, "n": 0, "r": 0.0}
    rk = _ranks([abs(v) for v in d]); tot = sum(rk); wp = sum(r for r, v in zip(rk, d) if v > 0)
    if n <= 18:
        st = min(wp, tot - wp); c = 0
        for s in itertools.product((0, 1), repeat=n):
            w = sum(r for r, b in zip(rk, s) if b)
            if min(w, tot - w) <= st + 1e-9: c += 1
        p = c / 2 ** n
    else:
        z = (wp - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24); p = math.erfc(abs(z) / math.sqrt(2))
    return {"p": min(1.0, p), "n": n, "r": (wp - (tot - wp)) / tot}
def mcnemar(a, b):
    nb = sum(x and not y for x, y in zip(a, b)); nc = sum(y and not x for x, y in zip(a, b)); n = nb + nc
    if n == 0: return {"p": 1.0, "b": nb, "c": nc}
    k = min(nb, nc); p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"p": min(1.0, p), "b": nb, "c": nc}
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); run = 0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out
def ms(xs): return {"mean": round(mean(xs), 3), "sd": round(stdev(xs), 3) if len(xs) > 1 else 0} if xs else {"mean": 0, "sd": 0}

def cluster_stats(x, y, groups, n=2000, seed=7):
    """Paper §13.2 for <8 clusters: cluster-bootstrap CI of the mean paired difference + leave-one-cluster-out range."""
    d = [a - b for a, b in zip(x, y)]; by = {}
    for g, v in zip(groups, d): by.setdefault(g, []).append(v)
    keys = list(by)
    def m(ks):
        vals = [v for k in ks for v in by[k]]; return sum(vals) / len(vals)
    rnd = random.Random(seed)
    boots = sorted(m([rnd.choice(keys) for _ in keys]) for _ in range(n))
    loco = [m([k for k in keys if k != o]) for o in keys] if len(keys) > 1 else []
    return {"clusters": len(keys), "mean_diff": round(m(keys), 3),
            "cluster_boot_ci95": [round(boots[int(.025 * n)], 3), round(boots[min(n - 1, int(.975 * n))], 3)],
            "leave_one_cluster_out": [round(min(loco), 3), round(max(loco), 3)] if loco else None}

def krippendorff_alpha_ordinal(units, cats=(1, 2, 3, 4, 5)):
    """units: list of per-item rating lists (one value per rater). Ordinal metric, paper target alpha >= 0.70."""
    o = {}
    for vals in units:
        m = len(vals)
        if m < 2: continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j: o[(a, b)] = o.get((a, b), 0) + 1 / (m - 1)
    nc = {c: sum(o.get((c, k), 0) for k in cats) for c in cats}; n = sum(nc.values())
    if n <= 1: return float("nan")
    def d2(c, k):
        if c == k: return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        s = sum(nc[g] for g in cats if lo <= g <= hi) - (nc[lo] + nc[hi]) / 2
        return s * s
    do = sum(o.get((c, k), 0) * d2(c, k) for c in cats for k in cats) / n
    de = sum(nc[c] * nc[k] * d2(c, k) for c in cats for k in cats) / (n * (n - 1))
    return 1 - do / de if de else float("nan")