import importlib
import os
import re
import threading
import time

import pytest

from app.chain import (
    AsyncWriter,
    LogStore,
    merkle_path,
    merkle_root,
    validate_event,
    verify_path,
)
from app.evala import data as D
from app.evala import llm as L
from app.evala import metrics as M
from app.evala import run as R
from app.experiments import make_event


def ev(i=0, **kw): return {"event_type": "inference", "subject": f"user-{i}", "input": f"in {i}", "output": f"out {i}", **kw}
def types(st): return {i["type"] for i in st.verify()["issues"]}

@pytest.fixture
def s(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([make_event(i) for i in range(30)]); st.checkpoint(); return st

# ---- tamper state / restore / privacy ----
def test_tampered_flag_survives_restart(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([make_event(i) for i in range(10)]); st.checkpoint()
    st.tamper("delete", 5); st.close()
    st2 = LogStore(str(tmp_path)); assert st2.tampered and not st2.verify()["ok"]
    st2.restore(); assert not st2.tampered and st2.verify()["ok"]
def test_append_and_checkpoint_blocked_while_tampered(s):
    s.tamper("modify", 3)
    with pytest.raises(RuntimeError): s.append_batch([ev()])
    with pytest.raises(RuntimeError): s.checkpoint()
def test_append_after_restore_stays_valid(s):
    s.tamper("delete", 5); s.restore(); s.append_batch([ev(1)]); s.checkpoint(); assert s.verify()["ok"]
def test_unknown_tamper_kind_leaves_state_clean(s):
    with pytest.raises(ValueError): s.tamper("bogus")
    assert not s.tampered and not os.path.exists(s.bak_path)
def test_backup_wiped_after_restore(s):
    s.tamper("modify", 2); assert os.path.exists(s.bak_path); s.restore(); assert not os.path.exists(s.bak_path)
def test_shred_blocked_while_tampered(s):
    s.tamper("modify", 2)
    with pytest.raises(RuntimeError): s.shred("user-1")
def test_shredded_key_not_recoverable_via_tamper_restore(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([ev(1)]); st.checkpoint()
    sub = st.list(1)[0]["subject"]; st.shred(sub)
    st.tamper("modify", 1); st.restore()
    assert st.read_pii(1) is None and st.verify()["ok"]
def test_subject_is_pseudonymised_and_raw_id_resolves(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([ev(1, subject="alice")])
    assert st.db.execute("SELECT subject FROM events").fetchone()[0] != "alice"
    assert st.list(1)[0]["subject"].startswith("p_") and st.read_pii(1) is not None
    assert st.shred("alice") == 1 and st.read_pii(1) is None and st.pii_status(1) == "key destroyed"
def test_pii_not_in_envelope_columns(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([ev(1, input="secret-text")])
    row = st.db.execute("SELECT * FROM events").fetchone()
    assert "secret-text" not in " ".join(str(x) for x in row)
@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_secret_files_private(tmp_path):
    LogStore(str(tmp_path))
    for f in ("signing.key", "salt.bin"): assert os.stat(tmp_path / f).st_mode & 0o077 == 0

# ---- verification ----
def test_missing_anchor_file_detected(s): os.remove(s.anchor_path); assert "anchor_missing" in types(s)
def test_corrupt_anchor_line_detected(s):
    with open(s.anchor_path, "a") as f: f.write("{not json\n")
    assert "anchor_log_corrupt" in types(s)
def test_checkpoint_gap_detected(tmp_path):
    st = LogStore(str(tmp_path))
    for _ in range(3): st.append_batch([make_event(i) for i in range(10)]); st.checkpoint()
    st.db.execute("DELETE FROM checkpoints WHERE id=2"); st.db.commit()
    t = types(st); assert "checkpoint_gap" in t and "anchor_orphan" in t
def test_forged_insert_detected(s): s.tamper("insert"); assert "bad_signature" in types(s)
def test_replay_detected(s): s.tamper("replay", 5); assert "hash_mismatch" in types(s)
def test_int_timestamp_verifies(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([ev(1, ts=int(time.time()) - 10)]); assert st.verify()["ok"]
def test_merkle_proof_valid_and_detects_leaf_change(s):
    p = s.proof(5); assert p["valid"] and p["root"]
    s.db.execute("UPDATE events SET entry_hash=? WHERE seq=5", ("a" * 64,)); s.db.commit()
    assert s.proof(5)["valid"] is False
def test_merkle_path_odd_and_single():
    hs = [f"{i:064x}" for i in range(5)]; r = merkle_root(hs)
    assert all(verify_path(hs[i], merkle_path(hs, i), r) for i in range(5))
    assert verify_path(hs[0], merkle_path(hs[:1], 0), merkle_root(hs[:1]))
def test_concurrent_append_and_verify(tmp_path):
    st = LogStore(str(tmp_path)); errs = []
    def w(k):
        try:
            for i in range(25): st.append_batch([make_event(k * 100 + i)])
        except Exception as e: errs.append(e)
    ts = [threading.Thread(target=w, args=(k,)) for k in range(4)]
    for t in ts: t.start()
    for _ in range(10): st.verify()
    for t in ts: t.join()
    assert not errs and st.stats()["events"] == 100 and st.verify()["ok"]

# ---- event validation ----
@pytest.mark.parametrize("bad", [
    {"event_type": "nope"}, {"event_type": ["x"]}, ev(0, meta=[1]), ev(0, meta={"x": float("nan")}), ev(0, ts=time.time() + 9999),
    ev(0, subject="x" * 200), ev(0, input=5), ev(0, event_id="bad id!"), "not a dict"])
def test_invalid_events_rejected(bad):
    with pytest.raises(ValueError): validate_event(bad)
def bio(**over):
    now = time.time(); m = {"session_start": now - 5, "session_end": now, "reference_db": "db", "matched_input_ref": "img", "verifier_ids": ["a"]}
    m.update(over); return {"event_type": "biometric_match", "meta": m}
@pytest.mark.parametrize("over", [{"session_end": 1.0}, {"verifier_ids": []}, {"reference_db": " "}, {"matched_input_ref": 3}, {"session_start": "garbage"}])
def test_biometric_semantics_rejected(over):
    with pytest.raises(ValueError): validate_event(bio(**over))
def test_biometric_valid_accepted(s): s.append_batch([bio()]); assert s.verify()["ok"]
def test_duplicate_event_id_rejected_and_batch_atomic(s):
    n = s.stats()["events"]; s.append_batch([ev(1, event_id="e-1")])
    with pytest.raises(ValueError): s.append_batch([ev(2), ev(3, event_id="e-1")])
    assert s.stats()["events"] == n + 1
def test_retention_floor_still_enforced(s):
    with pytest.raises(ValueError): s.purge_pii(30)

# ---- async writer ----
def test_submit_validates_synchronously(tmp_path):
    w = AsyncWriter(LogStore(str(tmp_path)))
    with pytest.raises(ValueError): w.submit({"event_type": "nope"})
    w.close()
def test_submit_after_close_raises(tmp_path):
    w = AsyncWriter(LogStore(str(tmp_path))); w.close()
    with pytest.raises(RuntimeError): w.submit(ev())
def test_partial_failure_does_not_lose_good_events(tmp_path):
    st = LogStore(str(tmp_path)); w = AsyncWriter(st, policy="fail_closed")
    w.submit(ev(1, event_id="same")); w.submit(ev(2, event_id="same")); w.flush()
    assert st.stats()["events"] == 1 and w.failed == 1 and w.rejected; w.close()
def test_fail_open_records_gap_and_accounts_for_every_event(tmp_path):
    class Slow(LogStore):
        def append_batch(self, evs): time.sleep(0.05); return super().append_batch(evs)
    st = Slow(str(tmp_path)); w = AsyncWriter(st, batch=1, policy="fail_open", maxq=1)
    for i in range(20): w.submit(make_event(i))
    w.close()
    rows = [e for e in st.list(1000)]
    gaps = sum(e["meta"].get("dropped", 0) for e in rows if e["event_type"] == "logger_gap")
    normal = sum(e["event_type"] != "logger_gap" for e in rows)
    assert w.dropped > 0 and gaps == w.dropped and normal + w.dropped == 20 and st.verify()["ok"]

# ---- API ----
def mk(tmp_path, monkeypatch, key="k", demo="1"):
    monkeypatch.setenv("AUDIT_DATA_DIR", str(tmp_path / "d")); monkeypatch.setenv("AUDIT_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("AUDIT_DEMO", demo)
    if key: monkeypatch.setenv("AUDIT_API_KEY", key)
    else: monkeypatch.delenv("AUDIT_API_KEY", raising=False)
    import app.main as m; importlib.reload(m)
    from fastapi.testclient import TestClient
    return TestClient(m.app), m
H = {"X-API-Key": "k"}
def test_api_auth_and_validation(tmp_path, monkeypatch):
    c, m = mk(tmp_path, monkeypatch)
    assert c.post("/api/events", json={"event_type": "inference"}).status_code == 401
    assert c.post("/api/events", json={"event_type": "inference"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/api/events", json={"event_type": "inference"}, headers=H).status_code == 200
    assert c.post("/api/events", json={"event_type": "nope"}, headers=H).status_code == 422
    assert c.post("/api/events", json={"event_type": "biometric_match", "meta": {}}, headers=H).status_code == 422
    assert c.post("/api/events", json={"event_type": "inference", "event_id": "a"}, headers=H).status_code == 200
    assert c.post("/api/events", json={"event_type": "inference", "event_id": "a"}, headers=H).status_code == 422
    assert c.get("/api/events?limit=0").status_code == 422 and c.get("/api/verify").json()["ok"] is True
    assert c.get("/api/events/1/pii").status_code == 401 and c.get("/api/events/1/pii/status").status_code == 200
    m.store.close()
def test_api_demo_gating_and_tamper_flow(tmp_path, monkeypatch):
    c, m = mk(tmp_path, monkeypatch)
    c.post("/api/simulate?n=10", headers=H); c.post("/api/checkpoint", headers=H)
    assert c.post("/api/tamper", json={"kind": "bogus"}, headers=H).status_code == 422 and not m.store.tampered
    assert c.post("/api/tamper", json={"kind": "delete", "seq": 3}, headers=H).status_code == 200
    assert c.get("/api/verify").json()["ok"] is False
    assert c.post("/api/events", json={"event_type": "inference"}, headers=H).status_code == 409
    assert c.post("/api/restore", headers=H).status_code == 200 and c.get("/api/verify").json()["ok"] is True
    m.store.close()
    c2, m2 = mk(tmp_path / "x", monkeypatch, demo="0")
    assert c2.post("/api/tamper", json={"kind": "delete"}, headers=H).status_code == 403
    assert c2.get("/api/info").json()["demo_enabled"] is False
    m2.store.close()
def test_api_evala_missing(tmp_path, monkeypatch):
    c, m = mk(tmp_path, monkeypatch); assert c.get("/api/evala").json() == {"missing": True}; m.store.close()

# ---- evaluation data / metrics / pipeline ----
def test_dataset_consistent(): assert D.validate() == []
def test_artifact_ids_do_not_collide_with_conditions_or_paper_experiments():
    assert all(re.fullmatch(r"CTL\d+|TST\d+|EVD\d+", k) for k in D.INV)
    assert not set(D.INV) & set(R.CONDS) and not any(re.fullmatch(r"E\d+", k) for k in D.INV)
def test_validate_catches_dangling_link(monkeypatch):
    monkeypatch.setitem(D.GOLD_LINKS, "G01", ["CTL1", "TST01", "EVD99"]); assert any("EVD99" in p for p in D.validate())
def test_keycov_no_substring_false_positives():
    assert M.keycov("technology catalog", ["record|log"]) == 0 and M.keycov("records are logged", ["record|log"]) == 1
def test_norm_refs_word_boundary():
    assert M.norm_refs(["part 12"]) == set() and M.norm_refs(["Art. 12(3)", "Annex III"]) == {"A12(3)", "ANXIII"}
def test_full_chain_requires_every_gold_requirement():
    g = D.GOLD_LINKS
    split = {"G06a": [x for x in g["G06a"] if x[0] == "C"], "G06b": [x for x in g["G06b"] if x[0] != "C"]}
    assert M.score_links(split, "U06")[1] is False
    assert M.score_links({k: g[k] for k in ("G06a", "G06b")}, "U06")[1] is True
def test_sample_sd(): assert M.ms([1, 3])["sd"] == 1.414
def test_cluster_stats_shape():
    r = M.cluster_stats([1, 1, 0, 0], [0, 0, 0, 0], ["a", "a", "b", "b"]); assert r["clusters"] == 2 and r["mean_diff"] == 0.5 and len(r["leave_one_cluster_out"]) == 2
def test_krippendorff():
    assert M.krippendorff_alpha_ordinal([[1, 1], [3, 3], [5, 5]]) == pytest.approx(1.0)
    assert M.krippendorff_alpha_ordinal([[1, 5], [3, 1], [5, 3]]) < 0.5
def test_llm_output_sanitising(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    be = L.LLM("groq")
    monkeypatch.setattr(be, "_try", lambda u: '{"links":[["x"],"CTL1","CTL1",5,"nope"]}')
    assert be.link("r", {"CTL1": "d"}, None) == (["CTL1"], True)
    monkeypatch.setattr(be, "_try", lambda u: '{"requirements":[{"text":5},{"text":"ok"},"str"]}')
    assert be.derive("t", [], False)[0]["requirements"] == [{"text": "ok"}]
def test_atomic_write(tmp_path):
    p = tmp_path / "a.json"; R._atomic(str(p), "{}"); R._atomic(str(p), '{"x":1}')
    assert p.read_text() == '{"x":1}' and [f.name for f in tmp_path.iterdir()] == ["a.json"]
def test_run_aborts_when_llm_calls_fail(tmp_path, monkeypatch):
    class Dead:
        name, model, errors, temp_ok = "anthropic", "x", 99, True
        def derive(self, *a): return {"requirements": []}, False
        def link(self, *a): return [], False
    monkeypatch.setattr(R, "make_backend", lambda k, m=None: Dead())
    with pytest.raises(RuntimeError): R.run("anthropic", 1, None, str(tmp_path), log=lambda m: None)
    assert not (tmp_path / "eval_a.json").exists()
def test_rating_sheet_has_no_clarity_and_results_have_cluster_stats(tmp_path):
    r = R.run("heuristic", 1, None, str(tmp_path), log=lambda m: None)
    head = (tmp_path / "rating_sheet.csv").read_text().splitlines()[0]
    assert "clarity" not in head and "class_TG_ID_US" in head
    assert r["stats"]["cluster_analysis"]["wilcoxon_valid_per_paper"] is False and r["meta"]["n_clusters"] == 4
def test_trace_matrix_tests_exist():
    from app.trace import matrix
    src = open(os.path.join(os.path.dirname(__file__), "test_chain.py"), encoding="utf-8").read()
    for row in matrix():
        for t in row["tests"]: assert f"def {t}(" in src