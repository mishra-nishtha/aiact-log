import pytest
from app.chain import LogStore, AsyncWriter, EVENT_TYPES, MIN_RETENTION_DAYS
from app.experiments import make_event

@pytest.fixture
def s(tmp_path):
    st = LogStore(str(tmp_path)); st.append_batch([make_event(i) for i in range(30)]); st.checkpoint(); return st

def types(st): return {i["type"] for i in st.verify()["issues"]}

def test_append_chain_valid(s): assert s.stats()["events"] == 30 and s.verify()["ok"]
def test_verify_clean_chain(s): v = s.verify(); assert v["ok"] and v["issues"] == []
def test_taxonomy_covers_art12():
    for t in ("inference", "risk_flag", "model_update", "pmm_signal", "operator_action", "biometric_match"): assert t in EVENT_TYPES
def test_event_fields_present(s):
    e = s.list(1)[0]
    for k in ("system_id", "model_version", "ts", "prev_hash", "entry_hash", "sig"): assert e[k]
def test_biometric_event_requires_12_3_fields(s):
    with pytest.raises(ValueError): s.append_batch([{"event_type": "biometric_match", "meta": {}}])
    assert s.stats()["events"] == 30
def test_retention_floor(s):
    with pytest.raises(ValueError): s.purge_pii(30)
    assert s.purge_pii(MIN_RETENTION_DAYS) == 0
def test_shred_keeps_chain_valid(s):
    e = s.list(1)[0]
    assert s.read_pii(e["seq"]) is not None
    s.shred(e["subject"])
    assert s.read_pii(e["seq"]) is None and s.verify()["ok"]
def test_modify_detected(s): s.tamper("modify", 10); assert "hash_mismatch" in types(s)
def test_rehash_forgery_detected(s): s.tamper("modify", 10, rehash=True); assert "bad_signature" in types(s)
def test_delete_detected(s): s.tamper("delete", 10); assert {"sequence_gap", "chain_break"} <= types(s)
def test_reorder_detected(s): s.tamper("reorder", 10); assert not s.verify()["ok"]
def test_backdate_detected(s): s.tamper("backdate", 10); assert "hash_mismatch" in types(s)
def test_truncate_detected_by_anchor(s): s.tamper("truncate", n=5); assert "tail_truncated" in types(s)
def test_restore_after_tamper(s): s.tamper("delete", 5); s.restore(); assert s.verify()["ok"]
def test_async_writer_flushes_all(tmp_path):
    st = LogStore(str(tmp_path)); w = AsyncWriter(st, policy="fail_closed")
    for i in range(300): w.submit(make_event(i))
    w.flush(); assert st.stats()["events"] == 300 and st.verify()["ok"]; w.close()
