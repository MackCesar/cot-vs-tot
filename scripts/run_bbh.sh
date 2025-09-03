#!/usr/bin/env bash
set -euo pipefail

# Run CoT on BIG-Bench Hard (date_understanding subset)
python -m src.cot \
  --dataset lukaemon/bbh --subset date_understanding --split test \
  --max-samples 20 --model openai:gpt-4o-mini \
  --outdir results

latest=$(ls -td results/cot_bbh_* | head -1)
python -m src.eval --run-dir "$latest"