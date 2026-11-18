import random
import tempfile
import time

from .chain import AsyncWriter, LogStore

TYPES = ["inference"] * 6 + ["risk_flag", "pmm_signal", "operator_action", "human_override", "model_update", "biometric_match"]

def make_event(i):
    rnd = random.Random(i)  # deterministic per index (reproducible runs)
    t = rnd.choice(TYPES)
    e = {"event_type": t, "subject": f"user-{i % 7}", "input": f"query {i} text", "output": f"result {i}",
         "model_version": "m-1.0", "meta": {"confidence": round(rnd.random(), 3)}}
    if t == "biometric_match":
        now = time.time()
        e["meta"] = {"session_start": now - 5, "session_end": now, "reference_db": "watchlist-v3",
                     "matched_input_ref": f"img-{i}", "verifier_ids": ["op-1", "op-2"]}
    return e

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p / 100))]

def summarize(lat_s):
    ms = [x * 1000 for x in lat_s]
    return {"p50_ms": round(pct(ms, 50), 4), "p95_ms": round(pct(ms, 95), 4), "p99_ms": round(pct(ms, 99), 4)}

def bench(n=500):
    evs = [make_event(i) for i in range(n)]
    res = {"n": n}
    lat = []
    for e in evs:
        t = time.perf_counter(); dict(e); lat.append(time.perf_counter() - t)
    res["no_logging"] = summarize(lat)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        s = LogStore(d); lat = []; t0 = time.perf_counter()
        for e in evs:
            t = time.perf_counter(); s.append_batch([e]); lat.append(time.perf_counter() - t)
        res["sync"] = {**summarize(lat), "events_per_s": round(n / (time.perf_counter() - t0))}
        s.close()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        s = LogStore(d); w = AsyncWriter(s, policy="fail_closed"); lat = []; t0 = time.perf_counter()
        for e in evs:
            t = time.perf_counter(); w.submit(e); lat.append(time.perf_counter() - t)
        w.flush()
        res["async"] = {**summarize(lat), "events_per_s": round(n / (time.perf_counter() - t0)),
                        "persisted": s.stats()["events"]}
        w.close(); s.close()
    return res

def run_attacks():
    cases = [("control (no tampering)", None, {}),
             ("modify an entry", "modify", {}),
             ("modify + recompute hash chain (no key)", "modify", {"rehash": True}),
             ("delete an entry", "delete", {}),
             ("reorder two entries", "reorder", {}),
             ("backdate a timestamp", "backdate", {}),
             ("forge an entry (no key)", "insert", {}),
             ("replay a copied entry", "replay", {}),
             ("truncate tail (covered by anchor)", "truncate", {"n": 5}),
             ("truncate tail after last anchor (KNOWN LIMIT)", "truncate", {"n": 5, "late": True})]
    out = []
    for name, kind, o in cases:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            s = LogStore(d)
            s.append_batch([make_event(i) for i in range(100)]); s.checkpoint()
            if o.get("late"): s.append_batch([make_event(i) for i in range(10)])
            if kind: s.tamper(kind, seq=50, n=o.get("n", 1), rehash=o.get("rehash", False))
            v = s.verify()
            out.append({"attack": name, "detected": not v["ok"],
                        "issues": sorted({i["type"] for i in v["issues"]}), "verify_ms": v["verify_ms"]})
            s.close()
    return out