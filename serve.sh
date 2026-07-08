#!/usr/bin/env bash
# ACTIVE: single Qwen3.5-4B (BF16) via vLLM on :8000, served as Qwen/Qwen3.5-4B-BF16.
# served-model-name = the FULL model name + quant so clients that store it (e.g.
# build_ai_corpus.py -> ai_corpus.db model_name) record the exact model+precision.
# NOTE: test_infer*.py still use model id "qwen3-vl" — update those if you point
# them here, or add "qwen3-vl" as a second --served-model-name alias.
# Runs in the vision conda env.
#
#   ./serve.sh                    # foreground (blocks)
#   ./serve.sh > vllm.log 2>&1 &  # background
#
# Stop:  pkill -9 -f vllm   (verify VRAM freed: nvidia-smi)
set -euo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # guard: cuda-nvcc activation
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"     #        expands these under set -u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vision
exec vllm serve "Qwen/Qwen3.5-4B" \
  --served-model-name "Qwen/Qwen3.5-4B-BF16" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-num-seqs 2 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels": 802816}' \
  --reasoning-parser qwen3 \
  --port 8000

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
