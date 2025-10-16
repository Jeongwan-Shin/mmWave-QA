#!/usr/bin/env bash

# Run ALL tasks (ActionQA, Movement, Count, Temporal, BodyMove)
# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --tasks all \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs

# Run ActionQA only
# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --tasks actionqa \
#   --options 4 \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs
#   --tasks temporal \

# Run Movement only (single aggregated file)
python /workspace/mmWave-QA/mmWave-QA_generation.py \
  --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
  --tasks movement \
  --output-format single \
  --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs

# Run Count only
# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --tasks count \
#   --count-comb-len 3 \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs

# Run Temporal only
# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --temporal-comb-len 3 \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs

# Run BodyMove only
# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --tasks bodymove \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs

# Seed example: add --seed 17 to any invocation

# python /workspace/mmWave-QA/mmWave-QA_generation.py \
#   --actions-json /workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json \
#   --tasks all \
#   --output-format single \
#   --output-root /workspace/mmWave-QA/mmWaveQA_benchmark/qa_outputs