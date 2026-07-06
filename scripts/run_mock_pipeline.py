"""Minimal mock pipeline for Chapter 5.

This script does not implement training. It validates the intended
data flow between planner, environment adapter, and policy modules.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "planner" / "task_graph_schema.json"


def load_task_context() -> dict:
    with SCHEMA.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_pipeline(payload: dict) -> str:
    nodes = payload["task_graph"]["nodes"]
    priors = payload["role_priors"]
    hard_constraints = payload["hard_constraints"]
    reward = payload["reward_template"]
    return "\n".join(
        [
            "Chapter 5 mock pipeline summary",
            f"- task nodes: {len(nodes)}",
            f"- role priors: {len(priors)}",
            f"- hard constraints: {len(hard_constraints)}",
            "- reward template: "
            + ", ".join(f"{k}={v}" for k, v in reward.items()),
            "- next step: connect planner output to chapter4 environment wrapper",
        ]
    )


if __name__ == "__main__":
    payload = load_task_context()
    print(summarize_pipeline(payload))
