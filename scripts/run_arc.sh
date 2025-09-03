#!/usr/bin/env bash
set -euo pipefail

# Run CoT on ARC-Challenge
python -m src.cot \
  --dataset ai2_arc --subset ARC-Challenge --split test \
  --max-samples 20 --model openai:gpt-4o-mini \
  --outdir results

latest=$(ls -td results/cot_arc_* | head -1)
python -m src.eval --run-dir "$latest"