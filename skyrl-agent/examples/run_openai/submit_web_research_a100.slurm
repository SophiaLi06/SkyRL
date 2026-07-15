#!/bin/bash
#SBATCH --job-name=skyagent-web-research-a100
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

# Single-A100 smoke test for the web-research traffic-instrumentation pipeline.
#
# Why one GPU is enough: the paper's Section 5.1 setup needs 4 GPUs because it
# runs full GRPO *training* (policy + rollout colocated, gradients, optimizer
# state -- ~96GB+ just for Adam's optimizer states on an 8B model). We don't
# need any of that to validate the network-traffic instrumentation -- we just
# need the ReAct agent to actually call search_engine/web_browser against real
# APIs, which is pure inference. So this script serves the policy model with
# vLLM (weights + KV cache only, no optimizer state) and drives the agent with
# a trainer-free script that talks to it over plain HTTP
# (examples/run_openai/run_web_research_react.py). That fits comfortably on
# one A100, no multi-GPU cluster required.
#
# This does NOT use `uv run --isolated`. That resolves and builds every base
# dependency of skyrl_agent (openhands-ai, browsergym, swebench, tevatron, ...
# -- all irrelevant to this pipeline), which on this cluster's older
# glibc/CUDA-driver combination repeatedly failed to build from source or
# pulled mismatched CUDA versions. Instead this script builds one dedicated
# venv on scratch space with a pinned, validated torch/vllm combination, and
# reuses it on every subsequent run (only rebuilds if missing).
#
# Fill in --partition above for your cluster. Run from the skyrl-agent repo
# root (or anywhere -- the script cd's into the repo root itself), and make
# sure a `.env` file (copied from `.env.template`) with GOOGLE_SEARCH_KEY,
# JINA_API_KEYS, WEB_SUMMARY_API_BASE=http://localhost:8000/v1,
# WEB_SUMMARY_MODEL=Qwen/Qwen3-8B, and SKYAGENT_WEB_CACHE_DIR is present in
# the repo root -- and that the SKYRL_LD_LIBRARY_PATH_EXPORT/FI_PROVIDER/
# LD_PRELOAD lines in .env.template (written for a different, AWS-EFA-based
# cluster) are removed or commented out; they reference library paths that
# don't exist on this cluster's OS and will spam ld.so warnings otherwise.
# STEM_LLM_JUDGE_URL can be left unset -- rewards will just report 0.0, which
# is expected (this run is for traffic/pipeline validation, not benchmark
# scoring).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # -> skyrl-agent/ repo root

SCRATCH_BASE="${SCRATCH_BASE:-/n/netscratch/yu_lab/Lab/$USER}"
VENV="$SCRATCH_BASE/skyagent-venv"
PY="$VENV/bin/python"
MODEL="${WEB_RESEARCH_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"

mkdir -p "$SCRATCH_BASE"
export UV_PYTHON_PREFERENCE=only-managed
export UV_CACHE_DIR="$SCRATCH_BASE/uv-cache"
export HF_HOME="$SCRATCH_BASE/hf-cache"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

# --- One-time environment build (skipped if the venv already has a working vLLM) ---
if ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  echo "[setup] Building venv at $VENV (first run only, ~15-20 min)"
  uv python install 3.12
  uv venv --python 3.12 "$VENV"

  # torch+vllm: pinned, explicit CUDA backend -- NOT "auto" and NOT an
  # unconstrained `vllm`. This cluster's driver caps at CUDA 12.9; letting
  # resolution pick "latest" repeatedly resolved wheels needing CUDA 13
  # (which a 12.9 driver cannot run -- CUDA compat is backward-, not
  # forward-, compatible) or a GitHub-release wheel built for a newer glibc
  # than this cluster's OS provides. vllm==0.8.5 + cu124 is the pinned
  # combination validated end-to-end on this cluster.
  export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
  uv pip install --python "$PY" --no-config --torch-backend cu124 --compile-bytecode "vllm==0.8.5"

  # skyrl_agent itself with --no-deps, to skip its heavy/irrelevant base deps
  # (openhands-ai, browsergym, swebench, tevatron, pylcs, ...) which don't
  # build cleanly here and aren't needed for this pipeline. transformers is
  # pinned to vllm 0.8.5's own declared minimum (>=4.51.1) -- an unconstrained
  # install pulls a version too new for vllm 0.8.5's tokenizer code.
  uv pip install --python "$PY" --no-config --no-deps --editable .
  uv pip install --python "$PY" --no-config \
    "transformers==4.51.1" datasets pyarrow huggingface_hub omegaconf hydra-core \
    loguru aiohttp json5 jsonlines pandas requests openai litellm

  # flashinfer (optional vLLM sampling accelerator) fails to *load* on this
  # cluster's older libstdc++ (undefined symbol at import time). vLLM's own
  # fallback to a pure-PyTorch sampler only triggers on a clean ImportError,
  # not the OSError a partially-broken flashinfer install raises -- so it
  # must not be installed at all, not just be broken.
  uv pip uninstall --python "$PY" apache-tvm-ffi flashinfer-cubin flashinfer-python torch-c-dlpack-ext >/dev/null 2>&1 || true
else
  echo "[setup] Reusing existing venv at $VENV"
fi

# --- .env ---
if [ ! -f .env ]; then
  echo "[setup] .env not found. Copy .env.template to .env, fill in GOOGLE_SEARCH_KEY," >&2
  echo "        JINA_API_KEYS, WEB_SUMMARY_API_BASE=http://localhost:${PORT}/v1," >&2
  echo "        WEB_SUMMARY_MODEL=${MODEL}, SKYAGENT_WEB_CACHE_DIR, and remove the" >&2
  echo "        SKYRL_LD_LIBRARY_PATH_EXPORT/FI_PROVIDER/LD_PRELOAD lines before rerunning." >&2
  exit 1
fi
set -a; source .env; set +a
unset LD_PRELOAD FI_PROVIDER SKYRL_LD_LIBRARY_PATH_EXPORT 2>/dev/null || true

# --- Dataset ---
if [ ! -f data/DeepResearch-Data/hle_webthinker_converted.parquet ]; then
  echo "[setup] Fetching dataset"
  "$PY" data/deep_research.py --output-dir data/DeepResearch-Data
fi

# --- Serve vLLM ---
export OPENAI_API_KEY=sc
export NET_METRICS_LOG_FILE="${NET_METRICS_LOG_FILE:-$PWD/net_metrics_${SLURM_JOB_ID:-local}.jsonl}"

echo "[run] serving ${MODEL} on port ${PORT}"
"$VENV/bin/vllm" serve "${MODEL}" \
  --port "${PORT}" --max-model-len 32768 --gpu-memory-utilization 0.90 \
  > "vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[run] shutting down vllm (pid ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[run] waiting for vllm server to come up..."
ready=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [ "${ready}" -ne 1 ]; then
  echo "[run] vllm server did not become ready in time; see vllm_${SLURM_JOB_ID:-local}.log" >&2
  exit 1
fi

echo "[run] generating trajectories"
"$PY" examples/run_openai/run_web_research_react.py

echo "[run] done. metrics at ${NET_METRICS_LOG_FILE}"
