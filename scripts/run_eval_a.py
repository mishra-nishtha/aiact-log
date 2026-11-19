"""Evaluation A: python scripts/run_eval_a.py --backend heuristic|groq [--model M] [--runs 3] [--dev]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.evala.run import run

a = argparse.ArgumentParser(); a.add_argument("--backend", default="heuristic"); a.add_argument("--model"); a.add_argument("--runs", type=int, default=1); a.add_argument("--dev", action="store_true")
x = a.parse_args(); r = run(x.backend, x.runs, x.model, "results", log=lambda s: print(s, end="\r"), dev=x.dev)
if x.dev: print(json.dumps(r, indent=1)[:1500]); sys.exit()
print(); print(json.dumps({k: r[k] for k in ("meta", "conditions", "schema_c4", "trace", "stats")}, indent=1)[:4000]); print("saved results/eval_a.json, rating_sheet.csv, rating_key.json")
