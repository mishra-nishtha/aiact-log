import re, math, random, itertools
from statistics import mean, pstdev
from .data import UNITS, GOLD, GOLD_LINKS, CORPUS
from .llm import toks

MATCH_T, GROUND_T = 0.6, 0.5
def keycov(text, keys):
    t = text.lower(); return sum(any(a in t for a in k.split("|")) for k in keys) / len(keys)
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
        for m in re.finditer(r"(?:article|art\.?)\s*(\d+)(?:\s*\((\d+)\))?", s): out.add(f"A{m.group(1)}" + (f"({m.group(2)})" if m.group(2) else ""))
        for m in re.finditer(r"annex\s+([ivx]+)", s): out.add("ANX" + m.group(1).upper())
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
    """pred: {gold_id: [artifact ids]} -> per-gold (tp,fp,fn) and full-chain flag for the unit."""
    per, cover = {}, set()
    for gid, *_ in GOLD[unit_id]:
        G, P = set(GOLD_LINKS[gid]), set(pred.get(gid, []))
        per[gid] = (len(G & P), len(P - G), len(G - P)); cover |= {x[0] for x in G & P}
    return per, {"C", "T", "E"} <= cover

def prf(tuples):
    tp, fp, fn = (sum(t[i] for t in tuples) for i in range(3))
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0}
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
def ms(xs): return {"mean": round(mean(xs), 3), "sd": round(pstdev(xs), 3)} if xs else {"mean": 0, "sd": 0}
