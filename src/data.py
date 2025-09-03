import argparse
from typing import List, Any, Dict

from datasets import load_dataset
from .utils import Example


def normalize_gsm8k(a: str) -> str:
    import re
    m = re.findall(r"[-+]?\d*\.?\d+", a)
    return m[-1] if m else a.strip()


def load_examples(dataset: str, subset: str | None, split: str, max_samples: int | None) -> List[Example]:
    ds = load_dataset(dataset, subset) if subset else load_dataset(dataset)
    d = ds[split]
    rows = d.select(range(min(len(d), max_samples))) if max_samples else d
    out: List[Example] = []

    for r in rows:
        if dataset.startswith("openai/gsm8k"):
            out.append(Example(q=r["question"], a=normalize_gsm8k(r["answer"]), meta={"id": r.get("id")}))

        elif dataset.startswith("ai2_arc"):
            # ARC can expose choices either as a dict-of-lists or a list-of-dicts.
            q = r["question"]
            ans = r.get("answerKey", r.get("answer", ""))
            ch = r.get("choices", [])

            choices: List[str] = []
            if isinstance(ch, dict):
                texts = ch.get("text", [])
                labels = ch.get("label", [])
                if labels and len(labels) == len(texts):
                    choices = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
                else:
                    choices = list(texts)
            elif isinstance(ch, list):
                for c in ch:
                    lab = c.get("label", "")
                    txt = c.get("text", "")
                    choices.append(f"{lab}) {txt}" if lab else txt)
            else:
                choices = [str(ch)]

            out.append(Example(q=q + "\nChoices: " + " | ".join(choices), a=ans, meta={"choices": choices}))

        elif dataset.startswith("lukaemon/bbh"):
            out.append(Example(q=r["input"], a=r["target"], meta={}))

        else:
            q = r.get("question", r.get("input", str(r)))
            a = r.get("answer", r.get("target", ""))
            out.append(Example(q=q, a=a, meta={}))

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=20)
    args = ap.parse_args()

    exs = load_examples(args.dataset, args.subset, args.split, args.max_samples)
    print(f"Loaded {len(exs)} examples")
    for e in exs[:3]:
        print("Q:", e.q[:120].replace("\n", " "))
        print("A:", e.a)
        print()
