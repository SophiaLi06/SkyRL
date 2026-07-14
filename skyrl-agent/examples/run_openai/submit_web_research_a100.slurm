#!/bin/bash
#SBATCH --job-name=skyagent-web-research-a100
#SBATCH --partition=<FILL_IN_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

# Single-A100 smoke test: serve a policy model locally with vLLM, then drive
# WebResearchTask rollouts against it (real Serper/Jina traffic), with network
# traffic metrics logged to NET_METRICS_LOG_FILE for later analysis.
#
# Fill in --partition above for your cluster. Run from the skyrl-agent repo
# root (or anywhere -- the script cd's into the repo root itself), and make
# sure a `.env` file (copied from `.env.template`) with GOOGLE_SEARCH_KEY,
# JINA_API_KEYS, WEB_SUMMARY_API_BASE=http://localhost:8000/v1,
# WEB_SUMMARY_MODEL=Qwen/Qwen3-8B, and SKYAGENT_WEB_CACHE_DIR is present in
# the repo root. STEM_LLM_JUDGE_URL can be left unset -- rewards will just
# report 0.0, which is expected (this run is for traffic/pipeline validation,
# not benchmark scoring).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # -> skyrl-agent/ repo root

MODEL="${WEB_RESEARCH_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"

export OPENAI_API_KEY=sc
export NET_METRICS_LOG_FILE="${PWD}/net_metrics_${SLURM_JOB_ID:-local}.jsonl"

echo "[slurm] serving ${MODEL} on port ${PORT}"
uv run --isolated --env-file .env vllm serve "${MODEL}" \
  --port "${PORT}" --max-model-len 32768 --gpu-memory-utilization 0.90 \
  > "vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[slurm] shutting down vllm (pid ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[slurm] waiting for vllm server to come up..."
ready=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [ "${ready}" -ne 1 ]; then
  echo "[slurm] vllm server did not become ready in time; see vllm_${SLURM_JOB_ID:-local}.log"
  exit 1
fi

echo "[slurm] running trajectory generation"
uv run --isolated --env-file .env \
  python examples/run_openai/run_web_research_react.py

echo "[slurm] done. metrics at ${NET_METRICS_LOG_FILE}"
