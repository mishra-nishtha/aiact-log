# Paper ↔ code mapping and honest gaps

| Paper element | Where | Status |
|---|---|---|
| §4.2 obligation schema | `app/evala/data.py` (UNITS gold schema), C4 prompt in `llm.py` | done |
| §5.1 dataset (atomic obligations) | `data.py` UNITS (14), GOLD (15 reqs), GOLD_LINKS | **14 units, below 20-30 target; single author** |
| §5.1 dev set outside Arts. 12/19 | `DEV_UNITS`, `--dev` flag | format check only |
| §5.2 annotation / α | `score_ratings.py` (exact, within-1, QWK) | **independent annotators not done; Krippendorff α not implemented** |
| §5.3 retrieval | `llm.py::retrieve` (TF-IDF top-3, excludes target text) | **paper says FAISS + MiniLM; drop-in swap pending** |
| §5.3 extraction/derivation/trace recovery | `llm.py`, `run.py` | done (Anthropic / Ollama / heuristic) |
| §5.3 run config recorded | `eval_a.json` meta (model, temp, runs, prompt hash) | done |
| §5.4 conditions B2/C3/C4 | `run.py` | done; B1 = gold |
| §5.5 metrics | `metrics.py` | done (proxy for correctness/completeness; human sheet exported) |
| §5.6 blinded human evaluation | `rating_sheet.csv` + key | **needs raters** |
| §5.7 case study | the reference implementation itself (real components/tests/evidence) | **replaces AIGov/synthetic: update §5.7 text** |
| §5.8 stats (Wilcoxon, McNemar, Holm, bootstrap) | `metrics.py` | done; exploratory |
| RQ3 failure analysis | `rq3_by_characteristic`, `failures` | automatic by flag; thematic coding not done |
| Evaluation B (logging) | `app/chain.py`, `experiments.py`, tests | done; simulated anchor, single node |
| Lifecycle status, change propagation | (not evaluated in paper) | out of scope |

## Before you write results
1. Run a real LLM (`--backend anthropic|ollama --runs 3`); never report the `heuristic` rows as LLM results.
2. Verify the paraphrased provision texts against EUR-Lex.
3. Get ≥1 independent person to review the gold requirements and ratings; state the status in the paper.
