import argparse
from pathlib import Path

from .data import load_examples
from .models import ModelClient, ModelConfig
from .prompts import SYSTEM_REASONING, cot_prompt,voter_prompt
from .utils import now_ts, save_jsonl, set_seed, Example

def run_got(client: ModelClient, question:str, iters: int = 4):
    """
    Simple Graph-of-Thoughts baseline:
    - For each iteration, sample multiple CoT answers with different temperatures.
    - A voter prompt selects the best candidate per round.
    - After all rounds, another voter selects the overall final.
    """
    candidates = []
    for _ in range(iters):
        varients = []
        for temp in (0.0,0.4,0.8):
            resp = client.complete(
                cot_prompt(question),
                system=SYSTEM_REASONING,
                temperature=temp,
                max_tokens=400
            )
            varients.append(resp)

        vp = voter_prompt(question, [f"{i+1}) {v}" for i,v in enumerate(varients)])
        voted = client.complete(vp, system=SYSTEM_REASONING, temperature=0.2, max_tokens=200)
        candidates.append(voted)
    # Final vote accress round winner
    final = client.complete(
        voter_prompt(question, candidates),
        system=SYSTEM_REASONING,
        temperature=0.2,
        max_tokens=200
    )
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--subset", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    exs = load_examples(args.dataset, args.subset, args.split, args.max_samples)
    client = ModelClient(ModelConfig(model=args.model))

    rows = []
    for i,ex in enumerate(exs):
        final = run_got(client, ex.q, args.iters)
        rows.append({"idx":i,"question":ex.q,"gold":ex.a,"final":final})
    run_name = f"got_{Path(args.dataset).name}_{now_ts()}"
    outp = Path(args.outdir) / run_name / "predictions.jsonl"
    save_jsonl(outp, rows)
    print(f"Saved {len(rows)}predictions to {outp}")

if __name__ == "__main__":
    main()