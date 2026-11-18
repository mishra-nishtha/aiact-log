import re, os, pytest
from app.evala.data import UNITS, GOLD, GOLD_LINKS, INV, TESTS
from app.evala import metrics as M
from app.evala.llm import extract_json, retrieve
from app.evala.run import run

def test_inventory_tests_exist_in_suite():
    src = open(os.path.join(os.path.dirname(__file__), "test_chain.py"), encoding="utf-8").read()
    for t in TESTS: assert f"def {t}(" in src
def test_gold_complete_and_links_valid():
    assert set(GOLD) == {u["id"] for u in UNITS}
    ids = {g[0] for v in GOLD.values() for g in v}
    assert ids == set(GOLD_LINKS)
    for g, l in GOLD_LINKS.items():
        assert all(x in INV for x in l) and {x[0] for x in l} == {"C", "T", "E"}
def test_retrieval_returns_crossref():
    assert any("Art. 72" in t for t, _ in retrieve("post-market monitoring referred to in Article 72", 3))
def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1} and extract_json("no json") is None
def test_keycov_and_refs():
    assert M.keycov("record each event automatically", ["automatic", "record|log", "event"]) == 1
    assert M.norm_refs(["Article 79(1)", "Annex III"]) == {"A79(1)", "ANXIII"}
def test_stats():
    assert M.wilcoxon([1, 2, 3], [1, 2, 3])["p"] == 1.0
    assert M.wilcoxon([5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6])["p"] < 0.05
    assert M.mcnemar([1, 0], [1, 0])["p"] == 1.0 and M.mcnemar([1] * 8, [0] * 8)["p"] < 0.05
    h = M.holm([0.01, 0.04, 0.03]); assert all(x >= y for x, y in zip(h, [0.01, 0.04, 0.03])) and max(h) <= 1
def test_pipeline_end_to_end_heuristic(tmp_path):
    r = run("heuristic", 1, None, str(tmp_path), log=lambda m: None)
    assert r["meta"]["is_llm"] is False and set(r["conditions"]) == {"B2", "C3", "C4"} and set(r["trace"]) == {"C3", "C4"}
    assert (tmp_path / "rating_sheet.csv").exists() and 0 <= r["trace"]["C3"]["f1"] <= 1
