"""Structured cognitive priors for Chapter 5 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class CognitivePrior:
    reward_weights: Dict[str, float]
    preferred_method: str
    preferred_safety_mode: str
    priority_focus: str
    state_token: str


def load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_prior(schema: dict, scenario) -> CognitivePrior:
    base = dict(schema["reward_template"])

    if scenario.sea_state >= 3.5:
        base["safe"] += 0.35
        base["protect"] += 0.18
    if scenario.communication_quality <= 0.45:
        base["safe"] += 0.20
        base["verify"] += 0.12
    if scenario.target_speed >= 2.0:
        base["verify"] += 0.10
        base["protect"] += 0.12
    if scenario.difficulty == "easy":
        base["energy"] += 0.10
    if scenario.regime == "stress":
        base["safe"] += 0.25
        base["protect"] += 0.10

    if scenario.communication_quality <= 0.45 or scenario.regime == "stress":
        preferred_method = "behavior_distributed"
    elif scenario.sea_state <= 2.0 and scenario.communication_quality >= 0.82:
        preferred_method = "centralized_optimal"
    else:
        preferred_method = "homogeneous_distributed"

    if scenario.sea_state >= 3.2 or scenario.target_speed >= 2.2:
        preferred_safety_mode = "conservative"
    else:
        preferred_safety_mode = "balanced"

    if base["protect"] >= max(base["search"], base["verify"]):
        priority_focus = "protect"
    elif base["verify"] >= base["search"]:
        priority_focus = "verify"
    else:
        priority_focus = "search"

    sea_bin = "highsea" if scenario.sea_state >= 3.5 else "lowsea"
    comm_bin = "weakcomm" if scenario.communication_quality <= 0.5 else "strongcomm"
    speed_bin = "fasttar" if scenario.target_speed >= 2.0 else "slowtar"
    state_token = f"{scenario.regime}_{scenario.difficulty}_{sea_bin}_{comm_bin}_{speed_bin}_{priority_focus}"

    return CognitivePrior(
        reward_weights=base,
        preferred_method=preferred_method,
        preferred_safety_mode=preferred_safety_mode,
        priority_focus=priority_focus,
        state_token=state_token,
    )
