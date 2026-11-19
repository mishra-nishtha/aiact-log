import json
import os
import threading
from typing import Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chain import EVENT_TYPES, MIN_RETENTION_DAYS, LogStore
from .evala.classify import run_classification
from .evala.issues import IssueStore
from .evala.llm import groq_configured
from .evala.metrics import retrieval_metrics
from .evala.ratings import report as ratings_report
from .evala.reference import reference_labels, units_report
from .evala.run import run as eval_run
from .experiments import EXPERIMENTS, bench, experiments_meta, make_event, run_attacks
from .trace import matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("AUDIT_DATA_DIR") or os.path.join(ROOT, "data")
RES = os.environ.get("AUDIT_RESULTS_DIR") or os.path.join(ROOT, "results")
DEMO = os.environ.get("AUDIT_DEMO", "1").strip().lower() not in ("0", "false", "no", "off")
BACKENDS = ("heuristic", "groq")

store = LogStore(DATA_DIR)
ISSUES = IssueStore(os.path.join(RES, "issues.db"))
app = FastAPI(title="Art. 12 Tamper-Evident Audit Log")

def need_demo():
    if not DEMO:
        raise HTTPException(403, "demo endpoints are disabled (AUDIT_DEMO=0)")

W = []                     # no authentication
WD = [Depends(need_demo)]  # demo-only

def call(fn, *a):
    try: return fn(*a)
    except ValueError as ex: raise HTTPException(422, str(ex))
    except RuntimeError as ex: raise HTTPException(409, str(ex))  # e.g. log is in demo-tampered state

class Ev(BaseModel):
    event_type: str
    subject: str = "anon"
    input: str = ""
    output: str = ""
    model_version: str = "m-1.0"
    system_id: Optional[str] = None
    event_id: Optional[str] = None
    ts: Optional[Union[float, str]] = None
    meta: dict = {}

class Tam(BaseModel):
    kind: str
    seq: int = 1
    n: int = 1
    rehash: bool = False

class Cat(BaseModel):
    name: str
    description: str

class Iss(BaseModel):
    category_id: int
    unit_id: str
    cond: Optional[str] = None
    title: str
    detail: str = ""

@app.post("/api/events", dependencies=W)
def add(e: Ev): return {"seq": call(store.append_batch, [e.model_dump(exclude_none=True)])[0]}

@app.get("/api/events")
def events(limit: int = Query(50, ge=1, le=1000)): return store.list(limit)

@app.get("/api/events/{seq}/pii", dependencies=W)
def pii(seq: int):
    p = store.read_pii(seq)
    return p if p else {"error": "unavailable (shredded, purged or none)"}

@app.get("/api/events/{seq}/pii/status")
def pii_status(seq: int): return {"seq": seq, "status": store.pii_status(seq)}

@app.post("/api/simulate", dependencies=WD)
def simulate(n: int = 20):
    return {"added": len(call(store.append_batch, [make_event(i) for i in range(max(1, min(n, 2000)))]))}

@app.post("/api/checkpoint", dependencies=W)
def cp(): return call(store.checkpoint) or {"info": "nothing new to checkpoint"}

@app.get("/api/checkpoints")
def checkpoints(): return store.checkpoints()

@app.get("/api/proof/{seq}")
def proof(seq: int):
    p = call(store.proof, seq)
    if p is None: raise HTTPException(404, "event is not covered by a checkpoint yet")
    return p

@app.get("/api/verify")
def verify(): return store.verify()

@app.post("/api/tamper", dependencies=WD)
def tamper(t: Tam):
    call(store.tamper, t.kind, t.seq, t.n, t.rehash)
    return {"ok": True}

@app.post("/api/restore", dependencies=WD)
def restore():
    if not store.tampered: return {"info": "nothing to restore"}
    call(store.restore); return {"ok": True}

@app.post("/api/shred/{subject}", dependencies=W)
def shred(subject: str): return {"pii_rows_unrecoverable": call(store.shred, subject)}

@app.post("/api/purge", dependencies=W)
def purge(days: int = 200): return {"purged": call(store.purge_pii, days)}

@app.get("/api/bench", dependencies=[Depends(need_demo)])
def b(n: int = 500): return bench(max(1, min(n, 3000)))

@app.get("/api/attacks", dependencies=[Depends(need_demo)])
def attacks(): return run_attacks()

@app.get("/api/trace")
def trace(): return matrix()

@app.get("/api/info")
def info():
    return {**store.stats(), "taxonomy": EVENT_TYPES, "retention_floor_days": MIN_RETENTION_DAYS,
            "demo_enabled": DEMO,
            "legacy_raw_subject_rows": store.legacy_subjects(),
            "groq_configured": groq_configured(), "evala_backends": list(BACKENDS)}


EV = {"state": "idle", "msg": "", "err": None}
EV_LOCK = threading.Lock()

@app.get("/api/evala")
def evala():
    p = os.path.join(RES, "eval_a.json")
    if not os.path.exists(p): return {"missing": True}
    with open(p, encoding="utf-8") as f: return json.load(f)

@app.get("/api/evala/status")
def evala_status(): return dict(EV)

@app.get("/api/evala/ratings")
def evala_ratings(): return call(ratings_report, RES)

@app.post("/api/evala/run", dependencies=WD)
def evala_run(backend: str = "heuristic", runs: int = 1, model: str = ""):
    if backend not in BACKENDS: raise HTTPException(422, "backend must be heuristic|groq")
    if backend == "groq" and not groq_configured(): raise HTTPException(422, "GROQ_API_KEY is not set on the server")
    with EV_LOCK:
        if EV["state"] == "running": raise HTTPException(409, "already running")
        EV.update(state="running", msg="starting", err=None)
    def job():
        try:
            eval_run(backend, min(max(runs, 1), 5), model or None, RES, log=lambda m: EV.update(msg=m))
            EV.update(state="done", msg="finished")
        except BaseException as ex: EV.update(state="error", err=str(ex) or type(ex).__name__)
    threading.Thread(target=job, daemon=True).start()
    return dict(EV)

# ---------- ledger, experiments E7-E10 ----------
@app.get("/api/ledger")
def ledger(): return store.ledger()

@app.get("/api/experiments", dependencies=[Depends(need_demo)])
def experiments(): return experiments_meta()

@app.get("/api/experiments/{eid}", dependencies=[Depends(need_demo)])
def experiment(eid: str):
    e = EXPERIMENTS.get(eid.upper())
    if e is None: raise HTTPException(404, "unknown experiment; use E7, E8, E9 or E10")
    return e["fn"]()

# ---------- Evaluation A additions ----------
@app.get("/api/evala/units")
def evala_units(): return units_report()

@app.get("/api/evala/reference-labels")
def evala_reference_labels(): return reference_labels()

@app.get("/api/evala/retrieval")
def evala_retrieval(): return retrieval_metrics()

@app.get("/api/evala/classification")
def evala_classification():
    p = os.path.join(RES, "eval_a_class.json")
    if not os.path.exists(p): return {"missing": True}
    with open(p, encoding="utf-8") as f: return json.load(f)

@app.post("/api/evala/classify", dependencies=WD)
def evala_classify(model: str = ""):
    if not groq_configured(): raise HTTPException(422, "GROQ_API_KEY is not set on the server")
    if not os.path.exists(os.path.join(RES, "eval_a_raw.json")): raise HTTPException(422, "run Evaluation A first (eval_a_raw.json is missing)")
    with EV_LOCK:
        if EV["state"] == "running": raise HTTPException(409, "already running")
        EV.update(state="running", msg="classification: starting", err=None)
    def job():
        try:
            run_classification(model or None, RES, log=lambda m: EV.update(msg=m))
            EV.update(state="done", msg="classification finished")
        except BaseException as ex: EV.update(state="error", err=str(ex) or type(ex).__name__)
    threading.Thread(target=job, daemon=True).start()
    return dict(EV)

# ---------- RQ3 issue categories ----------
@app.get("/api/issues/categories")
def issue_categories(): return ISSUES.categories()

@app.post("/api/issues/categories", dependencies=W)
def issue_category_add(c: Cat): return call(ISSUES.add_category, c.name, c.description)

@app.get("/api/issues")
def issues_list(category_id: Optional[int] = None): return ISSUES.issues(category_id)

@app.post("/api/issues", dependencies=W)
def issue_add(i: Iss): return call(ISSUES.add_issue, i.category_id, i.unit_id, i.cond, i.title, i.detail)

_static = os.path.join(ROOT, "static")
if os.path.isdir(_static): app.mount("/", StaticFiles(directory=_static, html=True), name="static")