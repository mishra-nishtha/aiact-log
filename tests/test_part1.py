import json
import time

import pytest

from app.chain import LogStore
from app.evala import reference
from app.evala.classify import decide
from app.evala.issues import IssueStore
from app.experiments import make_event, run_e7, run_e8, run_e9, run_e10

LEGAL = "Provision: High-risk AI systems shall technically allow for the automatic recording of events (logs) over the lifetime of the system."

def test_e7(): assert all(r["as_expected"] for r in run_e7())
def test_e9(): assert all(r["as_expected"] for r in run_e9())
def test_e10(): assert all(r["as_expected"] for r in run_e10())

def test_e8():
    rows = run_e8()
    assert all(r["accounted"] and r["chain_ok"] for r in rows)
    closed = next(r for r in rows if "fail-closed" in r["scenario"])
    assert closed["dropped"] == 0 and closed["persisted"] == closed["submitted"]
    down = rows[2]
    assert down["unrecorded"] == 200 and down["persisted"] == 50

def test_time_report_flags_past_timestamps(tmp_path):
    s = LogStore(str(tmp_path))
    s.append_batch([make_event(i) for i in range(5)])
    s.append_batch([{**make_event(9), "ts": time.time() - 3600}])
    assert s.time_report()["anomaly_count"] >= 1
    s.close()

def test_ledger_ok_then_broken(tmp_path):
    s = LogStore(str(tmp_path))
    s.append_batch([make_event(i) for i in range(10)]); s.checkpoint()
    s.append_batch([make_event(i) for i in range(10, 20)]); s.checkpoint()
    L = s.ledger()
    assert L["chain_ok"] and len(L["blocks"]) == 2
    s.close()
    p = tmp_path / "anchors.jsonl"
    lines = p.read_text().splitlines()
    a = json.loads(lines[0]); a["prev"] = "1" * 64; lines[0] = json.dumps(a)
    p.write_text("\n".join(lines) + "\n")
    s2 = LogStore(str(tmp_path))
    assert not s2.ledger()["chain_ok"]
    s2.close()

def test_reference_consistent(): assert reference.validate_reference() == []

def test_units_arithmetic():
    r = reference.units_report()
    assert r["counts"]["paper"] == 10 and r["counts"]["pilot"] == 14
    assert r["split"] == {"U1": ["U01", "U02"], "U10": ["U12", "U13", "U14"]} and r["pilot_only"] == ["U03"]

def test_decide_tg():
    assert decide({"cited_span": "automatic recording of events", "added_content": []}, LEGAL)["class"] == "TG"

def test_decide_uncited_is_us():
    assert decide({"cited_span": "tamper-proof storage required", "added_content": []}, LEGAL)["class"] == "US"

def test_decide_id_needs_full_justification():
    base = {"cited_span": "automatic recording of events", "added_content": ["hash chain"], "class": "TG"}
    d = decide(base, LEGAL)
    assert d["class"] == "US" and d["llm_class"] == "TG"          # model label recorded, not trusted
    full = {**base, "reasoning": "integrity serves traceability", "purpose": "detect tampering", "alternatives": ["WORM storage"]}
    assert decide(full, LEGAL)["class"] == "ID"

def test_issues(tmp_path):
    st = IssueStore(str(tmp_path / "i.db"))
    cats = st.categories()
    assert len(cats) == 5 and all(c["builtin"] for c in cats)
    c = st.add_category("Negation scope", "x")
    with pytest.raises(ValueError): st.add_category("negation SCOPE", "y")
    st.add_issue(c["id"], "U03", "C3", "first issue", "d")
    with pytest.raises(ValueError): st.add_issue(c["id"], "U99", None, "bad unit", "")
    assert st.issues()[0]["category"] == "Negation scope"