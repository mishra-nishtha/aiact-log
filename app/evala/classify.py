"""Stage-2 classification TG / ID / US (paper §9). Groq only. Separate from derive/link so frozen prompts are unchanged."""
import hashlib
import json
import os
import time
from statistics import mean

from .data import CORPUS, GOLD, UNITS
from .llm import SYS, extract_json, make_backend, toks
from .metrics import MATCH_T, keycov
from .reference import GOLD_CLASS, STATUS
from .run import _atomic

CONDS, CLASSES, MAX_FAIL_RATE = ("B2", "C3", "C4"), ("TG", "ID", "US"), 0.2

def legal_text(u):
    p = [f"Provision under analysis ({CORPUS[u['parent']][0]}): {CORPUS[u['parent']][1]}"]
    p += [f"Referenced provision ({CORPUS[r][0]}): {CORPUS[r][1]}" for r in u["refs"] if r != u["parent"]]
    return "\n".join(p)

def p_classify(legal, req):
    return (f"Legal text (the ONLY authority):\n{legal}\n\nCandidate technical requirement:\n{req}\n\n"
            "Apply this decision procedure.\n"
            "1. cited_span: copy VERBATIM the shortest span of the legal text that supports the requirement; \"\" if none does.\n"
            "2. added_content: list every mechanism, threshold, actor, scope or stricter condition in the requirement that the legal text does not state. "
            "[] if none (no implementation could satisfy the provision yet fail the requirement).\n"
            "3. Only if added_content is non-empty: reasoning (why it is a defensible reading), purpose (how it serves the provision's purpose), "
            "alternatives (other ways to satisfy the provision without it).\n"
            'Return JSON: {"cited_span":str,"added_content":[str],"reasoning":str,"purpose":str,"alternatives":[str],"class":"TG|ID|US"}')

CLASSIFY_PROMPT_HASH = hashlib.sha256((SYS + p_classify("l", "r")).encode()).hexdigest()[:12]

def _str(x): return x.strip() if isinstance(x, str) else ""
def _list(x): return [a.strip() for a in x if isinstance(a, str) and a.strip()] if isinstance(x, list) else []

def _span_found(span, legal):
    st, lt = toks(span), set(toks(legal))
    return len(st) >= 3 and sum(w in lt for w in st) / len(st) >= 0.8

def decide(j, legal):
    """Paper §9.4 applied to the model's structured answer. ID needs all of: cited span, added content named as a design
    choice, reasoning, purpose, alternatives; otherwise US. The model's own label is recorded as llm_class only."""
    if not isinstance(j, dict): return None
    span, added = _str(j.get("cited_span")), _list(j.get("added_content"))
    reasoning, purpose, alts = _str(j.get("reasoning")), _str(j.get("purpose")), _list(j.get("alternatives"))
    if not _span_found(span, legal): cls, why = "US", "no cited span that appears in the provision text"
    elif not added: cls, why = "TG", "span cited and nothing added"
    elif reasoning and purpose and alts: cls, why = "ID", "added content with reasoning, purpose and alternatives"
    else: cls, why = "US", "added content without reasoning, purpose and alternatives (paper §9.4 step 3)"
    lc = _str(j.get("class")).upper()
    return {"class": cls, "why": why, "cited_span": span, "added_content": added, "llm_class": lc if lc in CLASSES else None}

def _gold_match(unit_id, text):
    best, bs = None, 0.0
    for gid, _t, keys in GOLD[unit_id]:
        s = keycov(text, keys)
        if s > bs: best, bs = gid, s
    return best if bs >= MATCH_T else None

def run_classification(model, res_dir, log=print):
    raw_p = os.path.join(res_dir, "eval_a_raw.json")
    if not os.path.isfile(raw_p): raise ValueError("run Evaluation A first: results/eval_a_raw.json is missing")
    with open(raw_p, encoding="utf-8") as f: raw = json.load(f)["raw"]
    ev_meta, ep = {}, os.path.join(res_dir, "eval_a.json")
    if os.path.isfile(ep):
        with open(ep, encoding="utf-8") as f: ev_meta = json.load(f).get("meta", {})
    be = make_backend("groq", model)          # raises SystemExit if GROQ_API_KEY is missing
    items, bad = [], 0
    for c in CONDS:
        for u in UNITS:
            runs = (raw.get(c) or {}).get(u["id"]) or []
            if not runs: continue
            legal = legal_text(u)
            for r in runs[0].get("requirements", []):           # first run only, like the rating sheet
                text = r.get("text") if isinstance(r, dict) else None
                if not isinstance(text, str) or not text.strip(): continue
                d = decide(extract_json(be._try(p_classify(legal, text))), legal)
                bad += d is None
                gid = _gold_match(u["id"], text)
                items.append({"cond": c, "unit": u["id"], "requirement": text,
                              "class": d["class"] if d else None, "llm_class": d["llm_class"] if d else None,
                              "why": d["why"] if d else "no usable Groq response",
                              "cited_span": d["cited_span"] if d else "", "added_content": d["added_content"] if d else [],
                              "gold_id": gid, "gold_class": GOLD_CLASS[gid]["class"] if gid else None})
            log(f"classify {c} {u['id']}")
    total = len(items)
    if total == 0: raise ValueError("no requirements found in eval_a_raw.json")
    if bad / total > MAX_FAIL_RATE:
        raise RuntimeError(f"{bad}/{total} Groq calls failed or were unparseable ({be.errors} API errors); results NOT written")
    summary = {}
    for c in CONDS:
        its = [i for i in items if i["cond"] == c]
        cnt = {k: sum(i["class"] == k for i in its) for k in CLASSES}
        un = sum(i["class"] is None for i in its)
        ok = len(its) - un
        summary[c] = {"n": len(its), **cnt, "unclassified": un,
                      "us_rate": round(cnt["US"] / ok, 3) if ok else None,
                      "acceptable_min": round(cnt["TG"] / ok, 3) if ok else None,            # TG only
                      "acceptable_max": round((cnt["TG"] + cnt["ID"]) / ok, 3) if ok else None}  # + ID, validity unrated
    m = [i for i in items if i["class"] and i["gold_id"]]
    conf = {}
    for i in m:
        k = f"{i['gold_class']}->{i['class']}"; conf[k] = conf.get(k, 0) + 1
    res = {"meta": {"backend": "groq", "model": be.model, "date": time.strftime("%Y-%m-%d %H:%M"),
                    "classify_prompt_hash": CLASSIFY_PROMPT_HASH, "eval_a_date": ev_meta.get("date"),
                    "source_backend": ev_meta.get("backend"), "source_is_llm": ev_meta.get("is_llm"),
                    "n_items": total, "parse_failures": bad, "api_errors": be.errors, "first_run_only": True,
                    "model_label_differs_from_rule": sum(1 for i in items if i["llm_class"] and i["class"] and i["llm_class"] != i["class"]),
                    "labels_status": STATUS,
                    "acceptable_note": "acceptable_min = TG share; acceptable_max = TG+ID share. ID validity (valid/partly/invalid) needs human raters (paper §9.6) and is not rated here."},
           "summary": summary,
           "agreement": {"n_matched": len(m), "raw_agreement": round(mean(i["class"] == i["gold_class"] for i in m), 3) if m else None,
                         "confusion_gold_to_model": conf,
                         "note": "against author-drafted labels; exploratory; a requirement is matched to a gold requirement by the existing key-term rule"},
           "items": items}
    _atomic(os.path.join(res_dir, "eval_a_class.json"), json.dumps(res, indent=2))
    return res