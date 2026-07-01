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

# =============================================================================
# ACTIVE CONFIG: Qwen/Qwen3.5-4B (BF16, multimodal, thinking-by-default)
# =============================================================================
# Cached at ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B (8.8GB BF16
# weights, arch Qwen3_5ForConditionalGeneration + Qwen3_5MTP head). vLLM 0.23
# supports the arch. served-model-name kept as "qwen3-vl" so test_infer.py /
# test_infer_dense.py (MODEL="qwen3-vl") need NO edit.
#
# VRAM budget (12227 MiB card, gpu-util 0.90 -> 11004 MiB cap):
#   - weights (BF16 + MTP)                    ~9050 MiB
#   - activations + vision encoder + CUDA ctx  ~650 MiB
#   = KV pool                                 ~1300 MiB
# KV cost/token (32 layers, 4 KV heads, head_dim 256):
#   fp8 KV  = 2*32*4*256*1 = 64 KiB/tok  -> ~1300 MiB / 64 KiB ~= 20,800 tok
#   (fp16 KV would halve this to ~10K)
# So ctx ceiling is ~20K. Set 16384 for headroom vs the estimate; if the boot
# log ("Available KV cache memory") shows more free, bump toward 20K.
#
# reasoning-parser: qwen3 (official Qwen3.5 card recommends qwen3; thinking mode
#   emits <think>...</think>, same format the clients read via reasoning_content).
# NOT enabled from the official guide (memory/stability on 12GB):
#   --speculative-config qwen3_next_mtp : MTP draft head = extra draft KV+weights;
#       skip on tight VRAM (add later if boot log shows spare KV).
#   --tool-call-parser qwen3_coder      : no tool calling in this pipeline.
#   --max-num-seqs 8                    : guide value assumes fat AWQ KV pool;
#       one 16K seq ~= 1024 MiB fp8, so keep it at 2 here.
#   --swap-space 4                      : unrecognized by vllm 0.23 serve (guide
#       targets a newer vllm); removed or boot fails with "unrecognized argument".
# enforce-eager kept: CUDA graphs would eat ~1-2GB we don't have.
#
# Official sampling (thinking, general): temperature 1.0, top_p 0.95, top_k 20,
# presence_penalty 1.5 — set these client-side, not here.
MODEL="Qwen/Qwen3.5-4B"

exec vllm serve "$MODEL" \
  --served-model-name qwen3-vl \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-num-seqs 2 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels": 802816}' \
  --reasoning-parser qwen3 \
  --port 8000

# =============================================================================
# PREVIOUS CONFIG: Qwen3-VL-8B-Thinking-FP8 (kept for rollback)
# =============================================================================
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
# MODEL="Qwen/Qwen3-VL-8B-Thinking-FP8"
# exec vllm serve "$MODEL" \
#   --served-model-name qwen3-vl \
#   --max-model-len 3072 \
#   --gpu-memory-utilization 0.90 \
#   --enforce-eager \
#   --kv-cache-dtype fp8 \
#   --max-num-seqs 2 \
#   --limit-mm-per-prompt '{"image":1,"video":0}' \
#   --mm-processor-kwargs '{"max_pixels": 802816}' \
#   --reasoning-parser qwen3 \
#   --port 8000
