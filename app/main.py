import hmac
import json
import os
import threading
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chain import EVENT_TYPES, MIN_RETENTION_DAYS, LogStore
from .evala.run import run as eval_run
from .experiments import bench, make_event, run_attacks
from .trace import matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("AUDIT_DATA_DIR") or os.path.join(ROOT, "data")
RES = os.environ.get("AUDIT_RESULTS_DIR") or os.path.join(ROOT, "results")
API_KEY = os.environ.get("AUDIT_API_KEY") or ""   # empty = auth not required (unchanged default)
DEMO = os.environ.get("AUDIT_DEMO", "1") != "0"   # default on (unchanged default)

store = LogStore(DATA_DIR)
app = FastAPI(title="Art. 12 Tamper-Evident Audit Log")

def require_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key.encode(), API_KEY.encode())):
        raise HTTPException(401, "missing or invalid X-API-Key")

def require_demo():
    if not DEMO:
        raise HTTPException(403, "demo endpoints are disabled (AUDIT_DEMO=0)")

AUTH = [Depends(require_key)]
DEMO_ONLY = [Depends(require_key), Depends(require_demo)]

@app.exception_handler(RuntimeError)  # LogStore._require_clean(): log is in the demo-tampered state
async def _tampered(_: Request, ex: RuntimeError):
    return JSONResponse({"detail": str(ex)}, status_code=409)

class Ev(BaseModel):
    event_type: str
    subject: str = "anon"
    input: str = ""
    output: str = ""
    model_version: str = "m-1.0"
    meta: dict = {}
    event_id: Optional[str] = None

class Tam(BaseModel):
    kind: str
    seq: int = 1
    n: int = 1
    rehash: bool = False

@app.post("/api/events", dependencies=AUTH)
def add(e: Ev):
    try: return {"seq": store.append_batch([e.model_dump()])[0]}
    except ValueError as ex: raise HTTPException(422, str(ex))

@app.get("/api/events")
def events(limit: int = Query(50, ge=1, le=1000)): return store.list(limit)

@app.get("/api/events/{seq}/pii", dependencies=AUTH)
def pii(seq: int):
    p = store.read_pii(seq)
    return p if p else {"error": "unavailable (shredded, purged or none)"}

@app.get("/api/events/{seq}/pii/status")
def pii_status(seq: int): return {"seq": seq, "status": store.pii_status(seq)}

@app.post("/api/simulate", dependencies=DEMO_ONLY)
def simulate(n: int = 20):
    return {"added": len(store.append_batch([make_event(i) for i in range(min(n, 2000))]))}

@app.post("/api/checkpoint", dependencies=AUTH)
def cp(): return store.checkpoint() or {"info": "nothing new to checkpoint"}

@app.get("/api/checkpoints")
def checkpoints(): return store.checkpoints()

@app.get("/api/proof/{seq}")
def proof(seq: int):
    try: p = store.proof(seq)
    except ValueError as ex: raise HTTPException(409, str(ex))
    if p is None: raise HTTPException(404, "no checkpoint covers this event yet")
    return p

@app.get("/api/verify")
def verify(): return store.verify()

@app.post("/api/tamper", dependencies=DEMO_ONLY)
def tamper(t: Tam):
    try: store.tamper(t.kind, t.seq, t.n, t.rehash)
    except ValueError as ex: raise HTTPException(422, str(ex))
    return {"ok": True}

@app.post("/api/restore", dependencies=DEMO_ONLY)
def restore():
    if not store.tampered: return {"info": "nothing to restore"}
    try: store.restore()
    except ValueError as ex: raise HTTPException(409, str(ex))
    return {"ok": True}

@app.post("/api/shred/{subject}", dependencies=AUTH)
def shred(subject: str): return {"pii_rows_unrecoverable": store.shred(subject)}

@app.post("/api/purge", dependencies=AUTH)
def purge(days: int = 200):
    try: return {"purged": store.purge_pii(days)}
    except ValueError as ex: raise HTTPException(422, str(ex))

@app.get("/api/bench", dependencies=[Depends(require_demo)])
def b(n: int = 500): return bench(min(n, 3000))

@app.get("/api/attacks", dependencies=[Depends(require_demo)])
def attacks(): return run_attacks()

@app.get("/api/trace")
def trace(): return matrix()

@app.get("/api/info")
def info():
    return {**store.stats(), "taxonomy": EVENT_TYPES, "retention_floor_days": MIN_RETENTION_DAYS,
            "auth_required": bool(API_KEY), "demo_enabled": DEMO}

EV = {"state": "idle", "msg": "", "err": None}

@app.get("/api/evala")
def evala():
    p = os.path.join(RES, "eval_a.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"missing": True}

@app.get("/api/evala/status")
def evala_status(): return EV

@app.post("/api/evala/run", dependencies=AUTH)
def evala_run(backend: str = "heuristic", runs: int = 1, model: str = ""):
    if backend not in ("heuristic", "groq", "anthropic", "ollama"): raise HTTPException(422, "backend must be heuristic|groq|anthropic|ollama")
    if EV["state"] == "running": raise HTTPException(409, "already running")
    def job():
        EV.update(state="running", msg="starting", err=None)
        try:
            eval_run(backend, min(max(runs, 1), 5), model or None, RES, log=lambda m: EV.update(msg=m))
            EV.update(state="done", msg="finished")
        except BaseException as ex: EV.update(state="error", err=str(ex))
    threading.Thread(target=job, daemon=True).start(); return EV

app.mount("/", StaticFiles(directory=os.path.join(ROOT, "static"), html=True), name="static")