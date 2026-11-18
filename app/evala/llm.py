"""Retrieval, prompts and backends (heuristic baseline | anthropic | ollama)."""
import json, os, re, math, time, hashlib, urllib.request, urllib.error
from collections import Counter
from .data import CORPUS

STOP = set("the a an of to and or in for is are be shall that this with by on as at from it its which any each such than not may".split())
def toks(t): return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]

class TfIdf:
    def __init__(self, docs):  # docs: {id: text}
        self.ids = list(docs); self.tf = {i: Counter(toks(t)) for i, t in docs.items()}
        df = Counter(w for c in self.tf.values() for w in c)
        n = len(docs); self.idf = {w: math.log((n + 1) / (d + 0.5)) + 1 for w, d in df.items()}
    def _vec(self, c): return {w: f * self.idf.get(w, 1.0) for w, f in c.items()}
    def rank(self, q, exclude=()):
        qv = self._vec(Counter(toks(q))); qn = math.sqrt(sum(v * v for v in qv.values())) or 1
        out = []
        for i in self.ids:
            if i in exclude: continue
            dv = self._vec(self.tf[i]); dn = math.sqrt(sum(v * v for v in dv.values())) or 1
            out.append((i, sum(qv[w] * dv.get(w, 0) for w in qv) / (qn * dn)))
        return sorted(out, key=lambda x: -x[1])

_RET = TfIdf({k: v[1] for k, v in CORPUS.items()})
def retrieve(query, k=3, exclude=()):
    """Top-k corpus passages (TF-IDF cosine; the paper's FAISS + sentence-embedding retriever is a drop-in replacement)."""
    return [(CORPUS[i][0], CORPUS[i][1]) for i, s in _RET.rank(query, exclude)[:k] if s > 0]

# ---------------- prompts (frozen; PROMPT_HASH recorded in results) ----------------
SYS = "You are a requirements engineer for regulated AI systems. Reply with ONE JSON object and nothing else."
def p_derive(text, ctx, schema):
    s = f"Provision text:\n{text}\n"
    if ctx: s += "\nContext passages (authoritative text):\n" + "\n".join(f"- [{i}] {t}" for i, t in ctx) + "\n"
    s += "\nDerive testable, auditable technical requirements for the logging subsystem that satisfy ONLY what this provision demands. Do not add requirements the text does not support.\n"
    if schema:
        s += ('Return JSON: {"obligation":{"actor":str,"actor_status":"stated|not_stated","action":str,"conditions":[str],"exceptions":[str],'
              '"cross_references":[str],"temporal_constraints":[str]},"requirements":[{"type":"functional|non-functional","text":str,"testable":bool,"auditable":bool}]}')
    else: s += 'Return JSON: {"requirements":[{"text":str}]}'
    return s
def p_link(req, inv, oblig):
    s = f"Requirement:\n{req}\n"
    if oblig: s += "\nObligation (structured):\n" + json.dumps(oblig) + "\n"
    s += "\nArtifact inventory:\n" + "\n".join(f"{k}: {v}" for k, v in inv.items())
    s += '\n\nSelect the artifacts that implement (C*), verify (T*) or evidence (E*) this requirement. Return JSON: {"links":["C1","T01","E1"]}'
    return s
PROMPT_HASH = hashlib.sha256((SYS + p_derive("x", [("a", "b")], True) + p_derive("x", [], False) + p_link("r", {"C1": "d"}, {"a": 1})).encode()).hexdigest()[:12]

def extract_json(t):
    if not t: return None
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1]):
        try: return json.loads(cand)
        except Exception: pass
    return None

# ---------------- backends ----------------
class Heuristic:
    """Non-LLM rule-based baseline: sanity check of the evaluation harness. Its numbers are NOT LLM results."""
    name, model = "heuristic", "rule-based"
    def derive(self, text, ctx, schema):
        reqs = []
        for c in self._clauses(text):
            r = "The system shall ensure that " + c[0].lower() + c[1:] + "."
            if ctx and not reqs: r = r[:-1] + " (context: " + " ".join(ctx[0][1].split()[:16]) + ")."
            reqs.append({"text": r, "type": "functional", "testable": True, "auditable": True})
        out = {"requirements": reqs}
        if schema: out["obligation"] = self._oblig(text)
        return out, True
    @staticmethod
    def _clauses(t):
        parts = [p.strip(" ,.") for p in re.split(r";|\bunless\b|\bto the extent\b", t) if len(p.strip(" ,.")) > 12]
        return parts or [t]
    @staticmethod
    def _oblig(t):
        g = lambda p: [m.group(0).strip() for m in re.finditer(p, t, re.I)]
        return {"actor": "provider" if re.search(r"\bproviders?\b", t, re.I) else "", "actor_status": "stated" if re.search(r"\bproviders?\b|\bdeployers?\b", t, re.I) else "not_stated",
                "action": t.split(",")[0][:80], "conditions": g(r"to the extent[^,.;]*|appropriate to[^,.;]*|point 1\(a\) of Annex III"),
                "exceptions": g(r"unless[^.]*|without prejudice[^,.]*"), "cross_references": g(r"Article \d+(?:\(\d+\))?|Annex [IVX]+"),
                "temporal_constraints": g(r"over the lifetime[^,.;]*|at least [^,.;]*|for a period[^,.;]*")}
    def link(self, req, inv, oblig):
        q = req + (" " + json.dumps(oblig) if oblig else "")
        best = {}
        for i, s in TfIdf(inv).rank(q):
            if s > 0.04 and i[0] not in best: best[i[0]] = i
        return list(best.values()), True

class LLM:
    def __init__(self, kind, model=None):
        self.kind, self.name = kind, kind
        self.model = model or os.environ.get("LLM_MODEL") or ("claude-sonnet-5-5" if kind == "anthropic" else "llama3.1")
        self.temp_ok = True; self.errors = 0
        if kind == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"): raise SystemExit("Set ANTHROPIC_API_KEY (or use --backend heuristic|ollama)")
    def _post(self, url, headers, payload):
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
                with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf8", "ignore")
                if e.code == 400 and "temperature" in body and "temperature" in payload: payload.pop("temperature"); self.temp_ok = False; continue
                if e.code in (429, 500, 502, 503, 529): time.sleep(2 ** attempt * 2); continue
                raise RuntimeError(f"HTTP {e.code}: {body[:300]}")
            except urllib.error.URLError: time.sleep(2 ** attempt)
        raise RuntimeError("LLM request failed after retries")
    def _complete(self, user):
        if self.kind == "anthropic":
            r = self._post(os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
                           {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                           {"model": self.model, "max_tokens": 2000, "temperature": 0, "system": SYS, "messages": [{"role": "user", "content": user}]})
            return "".join(b.get("text", "") for b in r["content"])
        r = self._post("http://localhost:11434/api/chat", {"content-type": "application/json"},
                       {"model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
                        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]})
        return r["message"]["content"]
    def derive(self, text, ctx, schema):
        j = extract_json(self._try(p_derive(text, ctx, schema)))
        if not j or not isinstance(j.get("requirements"), list): return {"requirements": []}, False
        j["requirements"] = [r for r in j["requirements"] if isinstance(r, dict) and r.get("text")]
        return j, True
    def link(self, req, inv, oblig):
        j = extract_json(self._try(p_link(req, inv, oblig)))
        if not j or not isinstance(j.get("links"), list): return [], False
        return [x for x in j["links"] if x in inv], True
    def _try(self, u):
        try: return self._complete(u)
        except Exception as e: self.errors += 1; print("LLM error:", e); return ""

def make_backend(kind, model=None):
    return Heuristic() if kind == "heuristic" else LLM(kind, model)
