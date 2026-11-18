"""Run Evaluation-B experiments and save results to results/*.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.experiments import bench, run_attacks
os.makedirs("results", exist_ok=True)
a = run_attacks(); json.dump(a, open("results/attacks.json", "w"), indent=2)
print(f"{'attack':50} detected  issues")
for r in a: print(f"{r['attack']:50} {str(r['detected']):8}  {r['issues']}")
b = bench(500); json.dump(b, open("results/bench.json", "w"), indent=2)
print(json.dumps(b, indent=2))
