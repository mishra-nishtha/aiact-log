import json
import random
import tempfile
import time

from .chain import AsyncWriter, LogStore, canon, merkle_root

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

# ======================= E7-E10 (paper §10.9, §10.6) =======================
# All run on temporary stores; the live log is never touched. Rows share one shape:
# case, expected (detect | no_detect | known_limit), detected, layer, as_expected, issues, note.
ANCHOR_TYPES = {"anchor_mismatch", "tail_truncated", "anchor_checkpoint_mismatch", "anchor_missing",
                "anchor_orphan", "anchor_duplicate", "anchor_log_corrupt"}

def _row(case, expect, detected, issues=(), layer=None, note=""):
    t = set(issues)
    if layer is None:
        layer = "external anchor" if t & ANCHOR_TYPES else ("hash chain / signatures" if t else "none")
    return {"case": case, "expected": expect, "detected": bool(detected), "layer": layer if detected else "none",
            "as_expected": bool(detected) if expect == "detect" else not detected,
            "issues": sorted(t), "note": note}

def _base(d, n=100):
    s = LogStore(d)
    s.append_batch([make_event(i) for i in range(n)])
    s.checkpoint()
    return s

# ---------- E7: fork a node chain ----------
def _fork(s, at_seq, n_new):
    """Branch B: keep seq<=at_seq, drop the rest (and checkpoint rows covering it), append n_new different, validly
    signed events. anchors.jsonl (the witness) is left untouched."""
    with s.lock:
        for tbl in ("events", "pii", "event_ids"):
            s.db.execute(f"DELETE FROM {tbl} WHERE seq>?", (at_seq,))
        s.db.execute("DELETE FROM checkpoints WHERE to_seq>?", (at_seq,))
        s.db.commit()
        s._load_head()
    s.append_batch([{**make_event(10_000 + i), "input": f"branch-B {i}"} for i in range(n_new)])

def run_e7():
    out = []
    for name, expect, at, n_new, late in [
        ("control: single honest chain", "no_detect", None, 0, False),
        ("fork at seq 50: branch B re-written to seq 100 (anchor covers seq 100)", "detect", 50, 50, False),
        ("fork at seq 105, after the last anchor (KNOWN LIMIT)", "known_limit", 105, 5, True)]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            s = _base(d)
            if late: s.append_batch([make_event(200 + i) for i in range(10)])
            if at: _fork(s, at, n_new)
            v = s.verify()
            out.append(_row(name, expect, not v["ok"], {i["type"] for i in v["issues"]},
                            note=f"unanchored events: {v['unanchored_events']}"))
            s.close()
    return out

# ---------- E8: overload and logger failure ----------
def _e8_counts(s):
    gaps = [json.loads(m).get("dropped", 0) for (m,) in s.db.execute("SELECT meta FROM events WHERE event_type='logger_gap'")]
    persisted = s.db.execute("SELECT COUNT(*) FROM events WHERE event_type!='logger_gap'").fetchone()[0]
    return len(gaps), sum(gaps), persisted

def _e8_overload(policy, n):
    evs = [make_event(i) for i in range(n)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        s = LogStore(d); w = AsyncWriter(s, batch=32, policy=policy, maxq=8)  # tiny buffer forces overload
        accepted, lat = 0, []
        for e in evs:
            t = time.perf_counter(); accepted += bool(w.submit(e)); lat.append(time.perf_counter() - t)
        w.close(); st = w.stats()
        g_ev, g_tot, persisted = _e8_counts(s)
        v = s.verify(); s.close()
    note = ("dropped events are recorded as logger_gap entries" if policy == "fail_open"
            else "caller blocks when the buffer is full; nothing is dropped")
    return {"scenario": f"overload, {policy.replace('_', '-')}", "submitted": n, "accepted": accepted,
            "dropped": st["dropped"], "gap_events": g_ev, "gap_total": g_tot, "persisted": persisted,
            "unrecorded": st["failed"], "accounted": persisted + g_tot + st["failed"] == n,
            "chain_ok": v["ok"], "p99_ms": summarize(lat)["p99_ms"], "note": note}

def _e8_failure(n_down, n_up):
    evs = [make_event(i) for i in range(n_down + n_up)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        s = LogStore(d); w = AsyncWriter(s, batch=32, policy="fail_open", maxq=5000)
        def down(_batch): raise OSError("simulated storage failure")
        s.append_batch = down                      # injected failure: the store refuses every write
        for e in evs[:n_down]: w.submit(e)
        w.flush()
        del s.append_batch                         # store recovers
        for e in evs[n_down:]: w.submit(e)
        w.close(); st = w.stats()
        g_ev, g_tot, persisted = _e8_counts(s)
        v = s.verify(); s.close()
    return {"scenario": "logger failure, then recovery", "submitted": n_down + n_up, "accepted": n_down + n_up,
            "dropped": st["dropped"], "gap_events": g_ev, "gap_total": g_tot, "persisted": persisted,
            "unrecorded": st["failed"], "accounted": persisted + g_tot + st["failed"] == n_down + n_up,
            "chain_ok": v["ok"], "p99_ms": None,
            "note": "events lost while the store was down are counted only in the writer's counters; no logger_gap entry can be written while it refuses writes"}

def run_e8(): return [_e8_overload("fail_open", 3000), _e8_overload("fail_closed", 1000), _e8_failure(200, 50)]

# ---------- E9: writer-held signing key compromised after an anchored checkpoint ----------
def _forge_with_key(s, seq):
    """Attacker holds the signing key: change an entry, recompute hashes, RE-SIGN every entry and rebuild the checkpoints."""
    with s.lock:
        s.db.execute("UPDATE events SET output_hash=? WHERE seq=?", ("f" * 64, seq))
        s._rehash_from(seq)
        for sq, eh in s.db.execute("SELECT seq, entry_hash FROM events WHERE seq>=? ORDER BY seq", (seq,)).fetchall():
            s.db.execute("UPDATE events SET sig=? WHERE seq=?", (s.sk.sign(bytes.fromhex(eh)).hex(), sq))
        for cid, f, t in s.db.execute("SELECT id, from_seq, to_seq FROM checkpoints").fetchall():
            hs = [r[0] for r in s.db.execute("SELECT entry_hash FROM events WHERE seq BETWEEN ? AND ? ORDER BY seq", (f, t))]
            root, head = merkle_root(hs), hs[-1]
            sig = s.sk.sign(canon({"from": f, "to": t, "root": root, "head": head})).hex()
            s.db.execute("UPDATE checkpoints SET root=?, head_hash=?, sig=? WHERE id=?", (root, head, sig, cid))
        s.db.commit()
        s._load_head()

def run_e9():
    out = []
    for name, expect, mode in [
        ("control: key not compromised", "no_detect", None),
        ("key stolen: rewrite seq 50-100, re-sign entries and checkpoint (anchor covers seq 100)", "detect", "history"),
        ("key stolen: append forged events after the last anchor (KNOWN LIMIT)", "known_limit", "append")]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            s = _base(d)
            if mode == "history": _forge_with_key(s, 50)
            elif mode == "append":
                s.append_batch([{**make_event(300 + i), "system_id": "forged-by-key-holder"} for i in range(10)])
            v = s.verify()
            types = {i["type"] for i in v["issues"]}
            note = f"unanchored events: {v['unanchored_events']}"
            if mode == "history" and types and not (types - ANCHOR_TYPES):
                note = "hash chain, signatures and checkpoint all pass (attacker holds the key); only the witness differs"
            out.append(_row(name, expect, not v["ok"], types, note=note))
            s.close()
    return out

# ---------- E10: clock skew ----------
def run_e10():
    out = []
    for name, expect, skew, note in [
        ("control: no skew", "no_detect", 0, ""),
        ("caller clock +120 s (inside the 300 s tolerance)", "known_limit", 120, "accepted; inside tolerance"),
        ("caller clock +600 s (future)", "detect", 600, ""),
        ("caller clock -3600 s (events stamped one hour in the past)", "detect", -3600, "verify() does not inspect ts; the time check does")]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            s = LogStore(d)
            s.append_batch([make_event(i) for i in range(50)])
            rejected = False
            try: s.append_batch([{**make_event(500 + i), "ts": time.time() + skew} for i in range(10)])
            except ValueError: rejected = True
            s.checkpoint()
            v, t = s.verify(), s.time_report()
            s.close()
        if rejected: det, layer = True, "ingest validation (ts more than 300 s in the future)"
        elif not v["ok"]: det, layer = True, None
        elif t["anomaly_count"]: det, layer = True, "time-order check (host clock; not part of verify())"
        else: det, layer = False, None
        out.append(_row(name, expect, det, {i["type"] for i in v["issues"]} | {a["type"] for a in t["anomalies"]}, layer, note))
    return out

EXPERIMENTS = {
 "E7": {"title": "Fork a node chain", "paper": "§10.9 E7", "fn": run_e7,
        "summary": "Two histories branch from one node's chain; branch B is validly signed. The external witness commits to one head.",
        "limits": ["Single node: the paper's multi-node aggregator (§10.5) is not implemented, so one node's chain is forked.",
                   "The witness is the local file anchors.jsonl, not an independent third party.",
                   "A fork is detected only if an anchor covers the forked range; forks after the last anchor are not."]},
 "E8": {"title": "Overload the logger / logger failure", "paper": "§10.9 E8, §10.6, EO4", "fn": run_e8,
        "summary": "Fail-open vs fail-closed back-pressure under a deliberately small buffer, and a store that refuses writes then recovers.",
        "limits": ["Indicative, single node. The buffer is set to 8 events to force overload.",
                   "The paper does not declare either policy legally compliant (§10.6); neither does this experiment.",
                   "During a storage failure no gap entry can be written; the loss shows only in the writer's counters."]},
 "E9": {"title": "Compromise a writer-held signing key", "paper": "§10.9 E9", "fn": run_e9,
        "summary": "After an anchored checkpoint the attacker holds the signing key and forges history or appends forged events.",
        "limits": ["The paper assumes insiders lack signing keys (§10.9). Here the key is the file signing.key beside the database, so storage access implies key access.",
                   "Only history committed to the witness before the compromise is protected; later forged events are not detectable."]},
 "E10": {"title": "Inject clock skew", "paper": "§10.9 E10", "fn": run_e10,
         "summary": "Caller-supplied timestamps are skewed forward and backward; checked by ingest validation, verify() and a time-order check.",
         "limits": ["No RFC 3161 authority and no hybrid logical clock exist (paper §10.4), so trusted time is not established.",
                    "The time check compares entry times with checkpoint times, both from the host wall clock; a skewed host clock is not detected.",
                    "Tolerance 300 s, the same bound validate_event applies to future timestamps."]},
}
def experiments_meta():
    return [{"id": k, **{x: v[x] for x in ("title", "paper", "summary", "limits")}} for k, v in EXPERIMENTS.items()]