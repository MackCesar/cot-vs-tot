import argparse
import re
from pathlib import Path
from .data import load_examples
from .models import ModelClient, ModelConfig
from .prompts import SYSTEM_REASONING, cot_prompt
from .utils import now_ts, save_jsonl, set_seed

def extract_final(text: str) -> str:
    m = re.search(r"FINAL:\s*(.*)", text.strip(), flags=re.IGNORECASE)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--subset", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    exs = load_examples(args.dataset, args.subset, args.split, args.max_samples)
    client = ModelClient(ModelConfig(model=args.model))

    rows = []
    for i, ex in enumerate(exs):
        prompt = cot_prompt(ex.q)
        resp = client.complete(prompt, system=SYSTEM_REASONING, temperature=0.2, max_tokens=512)
        final = extract_final(resp)
        rows.append({"idx": i, "question": ex.q, "gold": ex.a, "response": resp, "final": final})

    run_name = f"cot_{Path(args.dataset).name}_{now_ts()}"
    outp = Path(args.outdir) / run_name / "predictions.jsonl"
    save_jsonl(outp, rows)
    print(f"Saved {len(rows)} predictions to {outp}")

if __name__ == "__main__":
    main()