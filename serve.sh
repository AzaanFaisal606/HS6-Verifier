#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Thinking-FP8 via vLLM (OpenAI-compatible API on :8000).
#
# Tuned for a 12GB RTX 5070: weights alone take ~10.18GB, so every flag below
# is what made the KV cache fit. Do not bump max-model-len above ~3072 without
# also freeing VRAM (smaller/quantized model or bigger GPU).
#
# Token budget per request (ctx = max-model-len = 3072):
#   ~1 vision token per 32x32 px  =>  tokens ~= pixels / 1024
#   max_pixels 802816 (896x896)   =>  784-token image ceiling
#   measured: a full-size image lands at ~760-785 prompt tokens.
#
#   | item                         | tokens |
#   |------------------------------|--------|
#   | max-size image               |  ~785  |
#   | chat template overhead       |  ~25   |
#   | input subtotal (image only)  |  ~810  |
#   | left for PROMPT + OUTPUT      | ~2260  |   <- shared budget below
#
#   Of that ~2260: subtract your prompt text, rest is generation.
#   This is a *Thinking* model — reasoning eats output first, so a long
#   prompt + a chatty reasoning trace can truncate the final answer.
#   Set max_tokens per request to reserve answer room. Rough token sizes:
#   ~1.18 tokens/word, ~5 chars/token. Smaller images free budget fast
#   (e.g. max_pixels 262144 -> ~256-token image, +530 freed).
#
# Usage:
#   ./serve.sh              # run in foreground (Ctrl-C to stop)
#   ./serve.sh > vllm.log 2>&1 &   # background, logs to file
set -euo pipefail

# --- activate the conda env that has vllm/torch/cuda ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vision

MODEL="Qwen/Qwen3-VL-8B-Thinking-FP8"

exec vllm serve "$MODEL" \
  --served-model-name qwen3-vl \
  --max-model-len 3072 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels": 802816}' \
  --reasoning-parser qwen3 \
  --port 8000
