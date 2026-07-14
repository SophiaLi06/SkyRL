"""Standalone, trainer-free trajectory generator for WebResearchTask.

Drives the ReAct agent (search_engine + web_browser tools) against a locally
served OpenAI-compatible model (e.g. `vllm serve Qwen/Qwen3-8B --port 8000`),
issuing real Serper/Jina traffic. No Ray, no gradient updates -- meant to run
on a single GPU purely to generate rollouts and network-traffic metrics for
studying rate limiting / shared caching.

Usage:
    uv run --isolated --env-file .env python examples/run_openai/run_web_research_react.py

Env vars:
    WEB_RESEARCH_DATASET   Path to a parquet file with a `prompt` column
                            (default: data/DeepResearch-Data/hle_webthinker_converted.parquet,
                            produced by `python data/deep_research.py`).
    WEB_RESEARCH_NUM_ROWS   How many rows to sample for this run (default: 8).
    WEB_RESEARCH_MODEL      Model name served by the local vLLM server
                            (default: Qwen/Qwen3-8B).
    NET_METRICS_LOG_FILE    Where to append traffic metrics JSONL. Defaults to
                            ./net_metrics.jsonl in the current directory if unset,
                            so this script produces useful data out of the box.
"""

import asyncio
import os
from pathlib import Path

import datasets
from transformers import AutoTokenizer

from skyrl_agent import AutoAgentRunner

os.environ["OPENAI_API_KEY"] = "sc"  # dummy key, assumes an unauth'ed vLLM service running locally
os.environ.setdefault("NET_METRICS_LOG_FILE", str(Path.cwd() / "net_metrics.jsonl"))

model = os.getenv("WEB_RESEARCH_MODEL", "Qwen/Qwen3-8B")
dataset_path = os.getenv(
    "WEB_RESEARCH_DATASET",
    str(Path(__file__).resolve().parents[2] / "data" / "DeepResearch-Data" / "hle_webthinker_converted.parquet"),
)
num_rows = int(os.getenv("WEB_RESEARCH_NUM_ROWS", "8"))

tokenizer = AutoTokenizer.from_pretrained(model)

dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
dataset = dataset.select(range(min(num_rows, len(dataset))))
print(f"Loaded {len(dataset)} rows from {dataset_path}")
print(dataset[0])

yaml_path = str(Path(__file__).parent / "web_research_react.yaml")

agent_generator = AutoAgentRunner.from_task(
    yaml_path,
    # no explicit inference engine with OpenAI -- built from backend_config in the yaml
    infer_engine=None,
    tokenizer=tokenizer,
)

output = asyncio.run(agent_generator.run(dataset, val_mode=True))
print("rewards:", output["rewards"])
print(f"Network traffic metrics written to: {os.environ['NET_METRICS_LOG_FILE']}")
