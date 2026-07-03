#!/usr/bin/env bash
# Serve a Qwen VLM via vLLM (OpenAI-compatible API on :8000).
#   ./serve.sh                    # foreground
#   ./serve.sh > vllm.log 2>&1 &  # background
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vision

# =============================================================================
# ACTIVE CONFIG: Qwen/Qwen3.5-4B (BF16) — 16384 ctx, fits the chapter-catalog
# prompt easily; no offload. This is the working box config for this machine.
# =============================================================================
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

# !! ALWAYS LAUNCH INSIDE THE CGROUP CAP. History:
#   - offload 12 UNCAPPED: host RAM spiked during mm-profiling, Windows RESET the
#     whole vmmem VM (killed SSH + tmux). HOST OOM = catastrophic.
#   - offload 9 CAPPED: on-GPU weights = 10.83GiB > the 0.90 util budget (10.75GiB)
#     -> GPU OOM at profiling. Only vllm died; VM survived. GPU OOM = safe.
#   Lesson: WSL sets pin_memory=False so the offload is pageable/swappable (NOT
#   pinned). The real failure point is GPU-at-profiling, and LESS offload puts
#   MORE weight on GPU -> worse. The cgroup cap makes host overruns kill vllm (not
#   the VM), so bias offload HIGHER for GPU headroom and let the cap guard host.
#   Launch:
#      tmux new-session -d -s vllm27 \
#        "systemd-run --user --scope -p MemoryMax=25G --unit=vllm27 \
#         bash -c './serve.sh 2>&1 | tee vllm.log'"
# Tuning (only after a clean, capped boot):
#   GPU OOM at profiling -> RAISE --cpu-offload-gb (less on GPU) and/or drop
#     max_pixels. on-GPU weights must sit well under gpu-mem-util*12227 to leave
#     room for KV + the vision-profiling forward (offload 12 -> ~7.84GiB on GPU).
#   host near the 25G cap -> the cap must stay < (WSL mem - ~2.5G baseline) or the
#     VM itself OOMs; with WSL memory=28GB that ceiling is ~25G. Don't exceed it.
#   need thinking room -> raise --max-model-len toward 8192 (4096 truncates the
#     chapter-gate prompt's thinking budget). Costs GPU KV -> go slow.

# =============================================================================
# ROLLBACK CONFIG: nvidia/Qwen3.6-27B-NVFP4 (CPU offload) — see 27b.md. Does NOT
# work on this box (no sm_120 native NVFP4 kernel; host-RAM balloon at profiling).
# Kept commented for reference only. ALWAYS launch inside the cgroup cap if tried.
# =============================================================================
# MODEL="nvidia/Qwen3.6-27B-NVFP4"
# exec vllm serve "$MODEL" \
#   --served-model-name qwen3-vl \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.90 \
#   --cpu-offload-gb 11 \
#   --enforce-eager \
#   --kv-cache-dtype fp8 \
#   --max-num-seqs 1 \
#   --tensor-parallel-size 1 \
#   --trust-remote-code \
#   --limit-mm-per-prompt '{"image":1,"video":0}' \
#   --mm-processor-kwargs '{"max_pixels": 200704}' \
#   --reasoning-parser qwen3 \
#   --port 8000

# =============================================================================
# CANDIDATE CONFIG: unsloth/Qwen3.6-27B-NVFP4 (25GB single shard) — superseded
# by nvidia's smaller 22GB build above. Kept for reference; offload 15.
# =============================================================================
# MODEL="unsloth/Qwen3.6-27B-NVFP4"
# exec vllm serve "$MODEL" \
#   --served-model-name qwen3-vl \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.88 \
#   --cpu-offload-gb 15 \
#   --enforce-eager \
#   --kv-cache-dtype fp8 \
#   --max-num-seqs 1 \
#   --tensor-parallel-size 1 \
#   --trust-remote-code \
#   --limit-mm-per-prompt '{"image":1,"video":0}' \
#   --mm-processor-kwargs '{"max_pixels": 802816}' \
#   --reasoning-parser qwen3 \
#   --port 8000

# =============================================================================
# PREVIOUS CONFIG: Qwen/Qwen3-VL-8B-Thinking-FP8
# =============================================================================
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
