"""Evaluation A orchestration: B2 (no retrieval) | C3 (retrieval) | C4 (retrieval + schema); trace-link recovery; stats; rating sheet.
NOTE: condition labels follow the pilot (B2/C3/C4), not the paper's RS/C0/C1/C1s/C2/C3 ladder (see review item A)."""
import csv
import io
import json
import os
import random
import time
from statistics import mean

from . import metrics as M
from .data import DEV_UNITS, GOLD, GOLD_LINKS, INV, UNITS
from .llm import PROMPT_HASH, make_backend, retrieve

CONDS = ("B2", "C3", "C4")
LABEL = {"B2": "Baseline 2: no retrieval, no schema", "C3": "Condition 3: retrieval", "C4": "Condition 4: retrieval + schema"}
MAX_FAIL_RATE = 0.2  # above this share of failed/unparseable LLM calls the run is aborted and nothing is written

def _atomic(path, text):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def derive_all(be, units, runs, log):
    raw = {c: {u["id"]: [] for u in units} for c in CONDS}; bad = 0
    for c in CONDS:
        for u in units:
            ctx = [] if c == "B2" else retrieve(u["text"], 3, exclude={u.get("parent", "")})
            for r in range(runs):
                out, ok = be.derive(u["text"], ctx, c == "C4"); bad += not ok
                raw[c][u["id"]].append({**out, "ok": ok, "ctx": [t for t, _ in ctx]})
            log(f"derive {c} {u['id']}")
    return raw, bad

def link_all(be, raw, runs, log):
    out = {c: {} for c in ("C3", "C4")}; bad = 0
    for c in out:
        for u in UNITS:
            for gid, text, _ in GOLD[u["id"]]:
                runs_l = []
                for r in range(runs):
                    ob = raw["C4"][u["id"]][r].get("obligation") if c == "C4" else None
                    l, ok = be.link(text, INV, ob); bad += not ok; runs_l.append(l)
                out[c][gid] = runs_l
            log(f"link {c} {u['id']}")
    return out, bad

def run(backend="heuristic", runs=1, model=None, out_dir="results", log=print, dev=False):
    t0 = time.time(); be = make_backend(backend, model); os.makedirs(out_dir, exist_ok=True)
    if dev:  # prompt-development check outside Arts. 12/19: format only, no scoring
        raw, bad = derive_all(be, DEV_UNITS, 1, log)
        return {"dev": True, "parse_failures": bad, "sample": raw["C4"]}
    runs = max(1, runs); raw, bad_d = derive_all(be, UNITS, runs, log)
    nclus = len({u["parent"] for u in UNITS})
    res = {"meta": {"backend": be.name, "model": be.model, "runs": runs,
                    "temperature": 0 if getattr(be, "temp_ok", True) else "provider default (temperature rejected)",
                    "prompt_hash": PROMPT_HASH, "date": time.strftime("%Y-%m-%d %H:%M"),
                    "n_units": len(UNITS), "n_clusters": nclus, "n_gold_reqs": len(GOLD_LINKS), "is_llm": backend != "heuristic",
                    "retriever": "tfidf-cosine top-3 (paper design: FAISS + sentence embeddings)", "text_source": "paraphrased provisions",
                    "gold_status": "author-drafted; independent annotation pending"}, "conditions": {}}
    per = {c: {} for c in CONDS}
    for c in CONDS:
        for u in UNITS:
            sc = [M.score_derive(u, [r["text"] for r in rr["requirements"]]) for rr in raw[c][u["id"]]]
            per[c][u["id"]] = {k: mean(s[k] for s in sc) for k in ("recall", "precision", "unsupported", "n")}
        res["conditions"][c] = {"label": LABEL[c], **{k: M.ms([per[c][u["id"]][k] for u in UNITS]) for k in ("recall", "precision", "unsupported", "n")}, "per_unit_recall": {u: round(v["recall"], 3) for u, v in per[c].items()}}
    sch = [s for u in UNITS for rr in raw["C4"][u["id"]] if (s := M.score_schema(u, rr.get("obligation")))]
    res["schema_c4"] = {k: round(mean(float(s[k]) for s in sch), 3) for k in ("xref_f1", "actor_ok", "temporal_ok", "exceptions_ok", "conditions_ok")} if sch else {}
    links, bad_l = link_all(be, raw, runs, log)
    total = len(CONDS) * len(UNITS) * runs + 2 * len(GOLD_LINKS) * runs; rate = (bad_d + bad_l) / total
    if backend != "heuristic" and rate > MAX_FAIL_RATE:
        raise RuntimeError(f"{bad_d + bad_l}/{total} LLM calls failed or returned unparseable output "
                           f"({getattr(be, 'errors', 0)} API errors); results NOT written")
    res["meta"].update({"parse_failures": bad_d + bad_l, "parse_failure_rate": round(rate, 3), "api_errors": getattr(be, "errors", 0)})
    res["trace"] = {}; fullbin = {}
    for c in links:
        tup, full = {g: [0, 0, 0] for g in GOLD_LINKS}, {u["id"]: [] for u in UNITS}; runs_prf = []
        for r in range(runs):
            tr = []
            for u in UNITS:
                pr = {g: links[c][g][r] for g, *_ in GOLD[u["id"]]}; per_g, ok = M.score_links(pr, u["id"]); full[u["id"]].append(ok)
                for g, t in per_g.items(): tr.append(t); tup[g] = [a + b / runs for a, b in zip(tup[g], t)]
            runs_prf.append(M.prf(tr))
        tl = [tuple(v) for v in tup.values()]
        fullbin[c] = {u: mean(v) >= 0.5 for u, v in full.items()}
        res["trace"][c] = {**{k: round(mean(p[k] for p in runs_prf), 3) for k in ("precision", "recall", "f1")}, "f1_ci95": [round(x, 3) for x in M.bootstrap_f1(tl)],
                           "completeness": round(mean(fullbin[c].values()), 3), "per_unit_full_chain": fullbin[c]}
    ids = [u["id"] for u in UNITS]; groups = [u["parent"] for u in UNITS]; rc = lambda c: [per[c][i]["recall"] for i in ids]
    w1, w2 = M.wilcoxon(rc("C3"), rc("B2")), M.wilcoxon(rc("C4"), rc("C3")); mc = M.mcnemar([fullbin["C3"][i] for i in ids], [fullbin["C4"][i] for i in ids])
    adj = M.holm([w1["p"], mc["p"], w2["p"]])
    res["stats"] = {"note": f"exploratory; n={len(UNITS)} non-independent units in {nclus} clusters; Holm-adjusted over 3 tests",
                    "RQ1_retrieval_B2_vs_C3_recall_wilcoxon": {**w1, "p_holm": adj[0]}, "RQ2_schema_C3_vs_C4_completeness_mcnemar": {**mc, "p_holm": adj[1]},
                    "extra_C3_vs_C4_recall_wilcoxon": {**w2, "p_holm": adj[2]},
                    "cluster_analysis": {"note": "paper §13.2: Wilcoxon only valid with >=8 clusters; below that use these cluster-level estimates",
                                         "n_clusters": nclus, "wilcoxon_valid_per_paper": nclus >= 8,
                                         "C3_minus_B2_recall": M.cluster_stats(rc("C3"), rc("B2"), groups),
                                         "C4_minus_C3_recall": M.cluster_stats(rc("C4"), rc("C3"), groups)},
                    "expert_gap_C4": {"recall_gap_to_gold": round(1 - res["conditions"]["C4"]["recall"]["mean"], 3), "note": "gold = maximum score; expert output is not separately rated"}}
    rq3 = {}
    for fl in ("crossref", "exception", "implicit_actor", "temporal", "condition"):
        w, wo = [u["id"] for u in UNITS if u["flags"][fl]], [u["id"] for u in UNITS if not u["flags"][fl]]
        rq3[fl] = {"n_with": len(w), "n_without": len(wo), **{c: {"recall_with": round(mean(per[c][i]["recall"] for i in w), 3) if w else None, "recall_without": round(mean(per[c][i]["recall"] for i in wo), 3) if wo else None} for c in CONDS}}
    res["rq3_by_characteristic"] = rq3
    miss = []
    for c in CONDS:
        for u in UNITS:
            if per[c][u["id"]]["recall"] < 1: miss.append({"cond": c, "unit": u["id"], "recall": round(per[c][u["id"]]["recall"], 2), "unsupported": round(per[c][u["id"]]["unsupported"], 2)})
    res["failures"] = miss; res["meta"]["seconds"] = round(time.time() - t0, 1)
    _atomic(os.path.join(out_dir, "eval_a_raw.json"), json.dumps({"raw": raw, "links": links}, indent=1))
    export_rating_sheet(raw, out_dir)
    _atomic(os.path.join(out_dir, "eval_a.json"), json.dumps(res, indent=2))  # written last: marks a complete run
    return res

def export_rating_sheet(raw, out_dir):
    """Paper §12.5: correctness + completeness (clarity is NOT rated); classification/failure columns left for raters."""
    items = [(c, u["id"], u["text"], r["text"]) for c in CONDS for u in UNITS for r in raw[c][u["id"]][0]["requirements"]]
    random.Random(11).shuffle(items); key = {}; buf = io.StringIO()
    w = csv.writer(buf); w.writerow(["item_id", "provision_unit_text", "requirement", "correctness_1to5", "completeness_1to5", "grounded_Y_N", "class_TG_ID_US", "failure_type"])
    for n, (c, uid, ut, rt) in enumerate(items, 1): w.writerow([f"R{n:03d}", ut, rt, "", "", "", "", ""]); key[f"R{n:03d}"] = {"cond": c, "unit": uid}
    _atomic(os.path.join(out_dir, "rating_key.json"), json.dumps(key, indent=1))
    _atomic(os.path.join(out_dir, "rating_sheet.csv"), buf.getvalue())