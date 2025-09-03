import argparse
import json
import re
from pathlib import Path

from .utils import load_jsonl

def normalize_num(s: str) -> str:
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    return m[-1] if m else s.strip()

def accuracy(rows):
    correct = 0
    total = 0
    for r in rows:
        gold = normalize_num(r.get("gold",""))
        pred = normalize_num(r.get("final",r.get("winner","")))
        if gold and pred and gold == pred:
            correct += 1
        total += 1
    return correct / max(total,1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    preds = load_jsonl(Path(args.run_dir) / "predictions.jsonl")
    acc = accuracy(preds)
    print(json.dumps({"metric":"accuracy", "value":acc, "n":len(preds)},indent=2))