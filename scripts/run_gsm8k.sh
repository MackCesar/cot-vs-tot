#!/usr/bin/env bash
set -euo pipefail

# Run Chain-of-Thought on GSM8K
python -m src.cot \
  --dataset openai/gsm8k --subset main --split test \
  --max-samples 20 --model openai:gpt-4o-mini \
  --outdir results

# Evaluate the last run
latest=$(ls -td results/cot_gsm8k_* | head -1)
python -m src.eval --run-dir "$latest"