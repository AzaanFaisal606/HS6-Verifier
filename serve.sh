#!/usr/bin/env bash
# =============================================================================
# ACTIVE: Qwen3.6-27B (GGUF UD-Q6_K_XL) via llama.cpp llama-server on :8001,
#         served-model-name (alias) "qwen3-vl" so test_infer*.py + testMatrix.py
#         need no model-id edit.
#
# This is the SAME 27B family as the vLLM NVFP4 config used for matrix Runs 7/8,
# but a different serving stack: unsloth dynamic Q6_K GGUF (24 GB weights) on
# llama.cpp instead of NVFP4 on vLLM. Weights do NOT fit the 12 GB card, so the
# offload is PARTIAL: --n-gpu-layers puts the first N transformer layers on the
# GPU and the remainder run on CPU (24 GB Q6_K vs ~11 GB usable VRAM => roughly a
# third of the net fits). The vision projector (mmproj, BF16, 889 MB) is kept on
# the GPU (--no-mmproj-offload would push it to CPU and make image encoding
# glacial). Expect SLOW generation — this is CPU-bound by design, same trade the
# NVFP4 + --cpu-offload-gb 12 config made.
#
# Tune --n-gpu-layers to taste: raise until nvidia-smi shows ~11 GB used, back off
# if llama-server OOMs at load. Host RAM holds the rest (~30 GB total, ~25 free).
#
#   ./serve.sh                          # foreground (blocks)
#   ./serve.sh > llama27b.log 2>&1 &    # background
#
# Stop:  pkill -f llama-server   (verify VRAM freed: nvidia-smi)
# =============================================================================
set -euo pipefail

LLAMA_DIR="$HOME/Desktop/llama-cuda"          # extracted keypaa sm120/cu12.8 build
LLAMA_SERVER="$LLAMA_DIR/bin/llama-server"    # binaries in bin/, libs in lib/
HF_SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf"
QWEN27B_GGUF="$HF_SNAP/Qwen3.6-27B-UD-Q6_K_XL.gguf"
QWEN27B_MMPROJ="$HF_SNAP/mmproj-BF16.gguf"

# keypaa build lacks CUDA runtime libs; vision env's pip nvidia-* provide them.
VENV="$HOME/miniforge3/envs/vision"
NV_LIBS="$(find "$VENV/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="$LLAMA_DIR/lib:$VENV/lib:$NV_LIBS${LD_LIBRARY_PATH:-}"

exec "$LLAMA_SERVER" \
  -m "$QWEN27B_GGUF" \
  --mmproj "$QWEN27B_MMPROJ" \
  --alias qwen3-vl \
  --port 8001 \
  --host 0.0.0.0 \
  --n-gpu-layers 22 \
  -c 8192 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0

# =============================================================================
# PRESERVED: single Qwen3.5-4B (BF16) via vLLM on :8000, served as
# Qwen/Qwen3.5-4B-BF16 (+ "qwen3-vl" alias).
# served-model-name = the FULL model name + quant so clients that store it (e.g.
# build_ai_corpus.py -> ai_corpus.db model_name) record the exact model+precision.
# Runs in the vision conda env. Stop: pkill -9 -f vllm
# =============================================================================
# export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # guard: cuda-nvcc activation
# export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"     #        expands these under set -u
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate vision
# exec vllm serve "Qwen/Qwen3.5-4B" \
#   --served-model-name "Qwen/Qwen3.5-4B-BF16" "qwen3-vl" \
#   --max-model-len 16384 \
#   --gpu-memory-utilization 0.90 \
#   --enforce-eager \
#   --kv-cache-dtype fp8 \
#   --max-num-seqs 2 \
#   --trust-remote-code \
#   --limit-mm-per-prompt '{"image":1,"video":0}' \
#   --mm-processor-kwargs '{"max_pixels": 802816}' \
#   --reasoning-parser qwen3 \
#   --port 8000

# =============================================================================
# PRESERVED: TWO-VLM ensemble via llama.cpp (llama-server, OpenAI-compatible).
#   InternVL3.5-4B  -> :8000  (served-model-name: internvl)
#   Qwen3.5-4B      -> :8001  (served-model-name: qwen)
# Both GGUF Q4_K_M + F16 mmproj (vision projector), all layers on GPU (-ngl 99).
# test_infer_ensemble.py fans out to both endpoints; RRF-fuses the two lists.
# To use it, comment out the vLLM block above and uncomment this whole block.
# Stop:  pkill -f llama-server   (verify VRAM freed: nvidia-smi)
# =============================================================================
# LLAMA_DIR="$HOME/Desktop/llama-cuda"          # extracted keypaa sm120/cu12.8 build
# LLAMA_SERVER="$LLAMA_DIR/bin/llama-server"    # binaries in bin/, libs in lib/
# MODELS="$HOME/Desktop/models"
#
# INTERNVL_GGUF="$MODELS/InternVL3_5-4B-GGUF/InternVL3_5-4B-Q4_K_M.gguf"
# INTERNVL_MMPROJ="$MODELS/InternVL3_5-4B-GGUF/mmproj-model-f16.gguf"
# QWEN_GGUF="$MODELS/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
# QWEN_MMPROJ="$MODELS/Qwen3.5-4B-GGUF/mmproj-F16.gguf"
#
# # keypaa build lacks CUDA runtime libs; vision env's pip nvidia-* provide them.
# VENV="$HOME/miniforge3/envs/vision"
# NV_LIBS="$(find "$VENV/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
# export LD_LIBRARY_PATH="$LLAMA_DIR/lib:$VENV/lib:$NV_LIBS${LD_LIBRARY_PATH:-}"
#
# COMMON=(--n-gpu-layers 99 -c 8192 --host 0.0.0.0 --flash-attn on)
#
# "$LLAMA_SERVER" -m "$INTERNVL_GGUF" --mmproj "$INTERNVL_MMPROJ" \
#   --alias internvl --port 8000 "${COMMON[@]}" &
# INTERNVL_PID=$!
# "$LLAMA_SERVER" -m "$QWEN_GGUF" --mmproj "$QWEN_MMPROJ" \
#   --alias qwen --port 8001 "${COMMON[@]}" &
# QWEN_PID=$!
# echo "internvl -> :8000 (pid $INTERNVL_PID)   qwen -> :8001 (pid $QWEN_PID)"
# echo "stop: pkill -f llama-server"
# wait
