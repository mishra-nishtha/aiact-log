# Art. 12 / 19(1) Tamper-Evident Audit Log: case-study implementation

Reference implementation for the paper *From Provision to Evidence*. Evaluation B (logging architecture) is implemented;
Evaluation A (LLM traceability pipeline) is a planned pilot, not included. The traceability matrix is builder-authored.

## Setup (Python 3.10+)
```bash
unzip aiact-log.zip && cd aiact-log
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Run
```bash
python -m uvicorn app.main:app --port 8000      # open http://localhost:8000
python -m pytest -q                             # 15 tests
python scripts/run_experiments.py               # attack suite + benchmark -> results/*.json
```
API docs: http://localhost:8000/docs

## 5-minute demo for your professor
1. **Simulate 20 events**, then **Checkpoint + anchor**, then **Verify** → green.
2. **Tamper**: choose "modify entry", seq 5, click Tamper, then Verify → red, row 5 highlighted.
3. Restore. Repeat with "modify" + *recompute hashes* → still caught (signature fails; attacker has no key).
4. Restore. "truncate tail" n=3 → caught by the anchor.
5. **Privacy**: click PII *view* on a row, **Crypto-shred** that subject, view again (gone), Verify (still green).
6. **Purge PII >30d** → refused (Art. 19(1) six-month floor).
7. **Run attack suite** and **benchmark**; show the traceability matrix at the bottom.

## Architecture
`app/chain.py`: hash chain (SHA-256) · Ed25519 signatures · Merkle checkpoints · anchor log (simulated witness) · per-subject Fernet keys (crypto-shredding) · retention floor · async bounded writer.
`app/experiments.py`: attack suite and benchmark · `app/trace.py`: matrix · `app/main.py`: FastAPI · `static/index.html`: dashboard.

## Honest limits (say these before the professor does)
- Hash chains **detect** tampering; they don't prevent it, and can't prove a logged event was true.
- Anchor is a **simulated** external witness (local file), not a real RFC 3161 TSA/transparency log. Not implemented: RFC 3161, distributed aggregation, hybrid logical clocks.
- Events after the last anchor are not protected against tail truncation (shown in the attack suite).
- Signing key sits on the same host (a compromised host can forge). Production: HSM/KMS.
- Single node, SQLite; benchmark numbers are indicative, not general.
- Art. 12 doesn't mandate tamper-evidence; it's an interpretive requirement (labelled so in the matrix).
- Tamper/restore endpoints are demo-only; remove them in any real deployment.

---
# Evaluation A: LLM-assisted pipeline pilot (added)
Pipeline: provision → (retrieval) → LLM → requirements (+ obligation schema) → trace links to this repo's real components / tests / endpoints.
Conditions: **B2** no retrieval · **C3** retrieval (top-3, incl. cross-referenced provisions) · **C4** retrieval + schema. 14 atomic Art. 12 / 19(1) units, 15 gold requirements.

## Run
```powershell
# 1) offline harness check (rule-based, NOT an LLM): works immediately
python scripts/run_eval_a.py --backend heuristic

# 2a) real LLM via Anthropic API (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:LLM_MODEL="claude-sonnet-5-5"        # optional; any model your key can use
python scripts/run_eval_a.py --backend anthropic --runs 3

# 2b) or free local model via Ollama (https://ollama.com; `ollama pull llama3.1`)
python scripts/run_eval_a.py --backend ollama --model llama3.1 --runs 3

# prompt-format check on the separate dev set (outside Arts. 12/19)
python scripts/run_eval_a.py --backend anthropic --dev
```
Outputs: `results/eval_a.json` (metrics, stats, RQ3), `results/eval_a_raw.json` (all LLM outputs), `results/rating_sheet.csv` + `rating_key.json` (blinded).
Refresh the dashboard: section 6 shows the results. A red banner marks non-LLM runs.

## Human ratings (paper §5.6)
1. Two raters each fill a copy of `results/rating_sheet.csv` (`rating_A.csv`, `rating_B.csv`; 1-5 scales, grounded Y/N).
2. `python scripts/score_ratings.py results/rating_A.csv results/rating_B.csv` → mean per condition, exact / within-1 agreement, quadratic-weighted kappa.

## Metrics (all automatic and reproducible; human ratings supersede)
- Recall/precision: a generated requirement matches a gold requirement if ≥60% of its key-term groups appear (keyword proxy, not semantic judgment).
- Unsupported rate: share of generated requirements with <50% of content words found in the provision + cross-referenced text.
- Trace links: P/R/F1 (bootstrap CI over gold requirements), traceability completeness (component + test + evidence all correct).
- Stats: exact Wilcoxon, exact McNemar, Holm over 3 tests. Exploratory (n=14, non-independent units).

See `PAPER_MAPPING.md` for the section-by-section mapping and the known gaps.
