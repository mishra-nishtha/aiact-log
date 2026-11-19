"""Retrieval, prompts and backends (heuristic baseline | groq | anthropic | ollama)."""
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from .data import CORPUS


def _load_env():
    """Load ROOT/.env (KEY=VALUE lines) without overriding real environment variables."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(root, ".env")
    if not os.path.isfile(p): return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1); k = k.strip()
            if k.startswith("export "): k = k[7:].strip()
            if k and k not in os.environ: os.environ[k] = v.strip().strip('"').strip("'")
_load_env()

def groq_configured(): return bool(os.environ.get("GROQ_API_KEY"))
class LLMFatal(RuntimeError):
    """Non-retryable provider error (bad key, unknown model, exhausted quota): abort the whole run."""

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


def retrieve_ids(query, k=5, exclude=()):
    """Top-k corpus ids (same TF-IDF ranking as retrieve), used for retrieval metrics."""
    return [i for i, s in _RET.rank(query, exclude)[:k] if s > 0]

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
    s += '\n\nSelect the artifacts that implement (CTL*), verify (TST*) or evidence (EVD*) this requirement. Return JSON: {"links":["CTL1","TST01","EVD1"]}'
    return s
PROMPT_HASH = hashlib.sha256((SYS + p_derive("x", [("a", "b")], True) + p_derive("x", [], False) + p_link("r", {"CTL1": "d"}, {"a": 1})).encode()).hexdigest()[:12]

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
    errors = 0
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
            if s > 0.04 and i[0] not in best: best[i[0]] = i  # best-scoring artifact per type (C/T/E)
        return list(best.values()), True

class LLM:
    """Groq (OpenAI-compatible chat completions). The only LLM provider used by this project."""
    URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    MAX_WAIT = 120  # a longer retry-after means a quota is exhausted: stop instead of sleeping

    def __init__(self, kind="groq", model=None):
        if kind != "groq": raise ValueError(f"unknown backend {kind!r} (only 'groq' is supported)")
        if not groq_configured(): raise SystemExit("Set GROQ_API_KEY (environment or .env), or use --backend heuristic")
        self.kind, self.name = "groq", "groq"
        self.model = model or os.environ.get("LLM_MODEL") or self.DEFAULT_MODEL
        self.temp_ok = True; self.errors = 0

    def _post(self, payload):
        headers = {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"], "content-type": "application/json",
                   "User-Agent": "aiact-log/1.0"}  # the default Python-urllib agent can be blocked with HTTP 403
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.URL, json.dumps(payload).encode(), headers)
                with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf8", "ignore")
                if e.code in (401, 403, 404): raise LLMFatal(f"Groq HTTP {e.code}: {body[:300]}")
                if e.code == 400 and "temperature" in body and "temperature" in payload:
                    payload.pop("temperature"); self.temp_ok = False; continue
                if e.code == 400 and ("response_format" in body or "json_validate_failed" in body) and "response_format" in payload:
                    payload.pop("response_format"); continue
                if e.code in (429, 500, 502, 503, 529):
                    try: wait = float(e.headers.get("retry-after", ""))
                    except (TypeError, ValueError): wait = 2 ** attempt * 2
                    if wait > self.MAX_WAIT: raise LLMFatal(f"Groq rate/quota limit: retry in {int(wait)} s")
                    time.sleep(max(wait, 1)); continue
                raise RuntimeError(f"HTTP {e.code}: {body[:300]}")
            except (urllib.error.URLError, TimeoutError): time.sleep(2 ** attempt)
        raise RuntimeError("Groq request failed after retries")

    def _complete(self, user):
        r = self._post({"model": self.model, "temperature": 0, "max_tokens": 2000,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]})
        return r["choices"][0]["message"].get("content") or ""

    def derive(self, text, ctx, schema):
        j = extract_json(self._try(p_derive(text, ctx, schema)))
        if not isinstance(j, dict) or not isinstance(j.get("requirements"), list): return {"requirements": []}, False
        j["requirements"] = [r for r in j["requirements"] if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip()]
        return j, True
    def link(self, req, inv, oblig):
        j = extract_json(self._try(p_link(req, inv, oblig)))
        if not isinstance(j, dict) or not isinstance(j.get("links"), list): return [], False
        return list(dict.fromkeys(x for x in j["links"] if isinstance(x, str) and x in inv)), True
    def _try(self, u):
        try: return self._complete(u)
        except LLMFatal: raise
        except Exception as e: self.errors += 1; print("LLM error:", e); return ""

def make_backend(kind, model=None):
    return Heuristic() if kind == "heuristic" else LLM(kind, model)