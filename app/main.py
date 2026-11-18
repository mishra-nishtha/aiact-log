import os, json, threading
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .chain import LogStore, EVENT_TYPES, MIN_RETENTION_DAYS
from .experiments import make_event, bench, run_attacks
from .trace import matrix
from .evala.run import run as eval_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
store = LogStore(os.path.join(ROOT, "data"))
app = FastAPI(title="Art. 12 Tamper-Evident Audit Log")

class Ev(BaseModel):
    event_type: str
    subject: str = "anon"
    input: str = ""
    output: str = ""
    model_version: str = "m-1.0"
    meta: dict = {}

class Tam(BaseModel):
    kind: str
    seq: int = 1
    n: int = 1
    rehash: bool = False

@app.post("/api/events")
def add(e: Ev):
    try: return {"seq": store.append_batch([e.model_dump()])[0]}
    except ValueError as ex: raise HTTPException(422, str(ex))

@app.get("/api/events")
def events(limit: int = 50): return store.list(limit)

@app.get("/api/events/{seq}/pii")
def pii(seq: int):
    p = store.read_pii(seq)
    return p if p else {"error": "unavailable (shredded, purged or none)"}

@app.post("/api/simulate")
def simulate(n: int = 20):
    return {"added": len(store.append_batch([make_event(i) for i in range(min(n, 2000))]))}

@app.post("/api/checkpoint")
def cp(): return store.checkpoint() or {"info": "nothing new to checkpoint"}

@app.get("/api/verify")
def verify(): return store.verify()

@app.post("/api/tamper")
def tamper(t: Tam):
    try: store.tamper(t.kind, t.seq, t.n, t.rehash)
    except ValueError as ex: raise HTTPException(422, str(ex))
    return {"ok": True}

@app.post("/api/restore")
def restore():
    if not store.tampered: return {"info": "nothing to restore"}
    store.restore(); return {"ok": True}

@app.post("/api/shred/{subject}")
def shred(subject: str): return {"pii_rows_unrecoverable": store.shred(subject)}

@app.post("/api/purge")
def purge(days: int = 200):
    try: return {"purged": store.purge_pii(days)}
    except ValueError as ex: raise HTTPException(422, str(ex))

@app.get("/api/bench")
def b(n: int = 500): return bench(min(n, 3000))

@app.get("/api/attacks")
def attacks(): return run_attacks()

@app.get("/api/trace")
def trace(): return matrix()

@app.get("/api/info")
def info(): return {**store.stats(), "taxonomy": EVENT_TYPES, "retention_floor_days": MIN_RETENTION_DAYS}

EV = {"state": "idle", "msg": "", "err": None}
RES = os.path.join(ROOT, "results")

@app.get("/api/evala")
def evala():
    p = os.path.join(RES, "eval_a.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"missing": True}

@app.get("/api/evala/status")
def evala_status(): return EV

@app.post("/api/evala/run")
def evala_run(backend: str = "heuristic", runs: int = 1, model: str = ""):
    if backend not in ("heuristic", "anthropic", "ollama"): raise HTTPException(422, "backend must be heuristic|anthropic|ollama")
    if EV["state"] == "running": raise HTTPException(409, "already running")
    def job():
        EV.update(state="running", msg="starting", err=None)
        try:
            eval_run(backend, min(max(runs, 1), 5), model or None, RES, log=lambda m: EV.update(msg=m))
            EV.update(state="done", msg="finished")
        except BaseException as ex: EV.update(state="error", err=str(ex))
    threading.Thread(target=job, daemon=True).start(); return EV

app.mount("/", StaticFiles(directory=os.path.join(ROOT, "static"), html=True), name="static")
