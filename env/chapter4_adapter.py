"""Adapter that turns the Chapter 4 simulator into a Chapter 5 RL environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ActionBundle:
    method: str
    safety_mode: str

    @property
    def label(self) -> str:
        return f"{self.method}|{self.safety_mode}"


ACTIONS: List[ActionBundle] = [
    ActionBundle("behavior_distributed", "balanced"),
    ActionBundle("behavior_distributed", "conservative"),
    ActionBundle("homogeneous_distributed", "balanced"),
    ActionBundle("homogeneous_distributed", "conservative"),
    ActionBundle("distance_only", "balanced"),
    ActionBundle("centralized_optimal", "balanced"),
]

COMPOSITE_SCORE_WEIGHTS = {
    "rescue_success": 130.0,
    "violation_rate": 18.0,
    "mission_time": 0.12,
    "formation_error": 0.35,
    "communication_load": 0.05,
}


def apply_safety_projection(metrics: Dict[str, float], scenario, safety_mode: str) -> Dict[str, float]:
    adjusted = dict(metrics)
    risk_index = (
        max(0.0, scenario.sea_state - 2.5)
        + 2.0 * max(0.0, 0.60 - scenario.communication_quality)
        + max(0.0, scenario.target_speed - 1.5)
    )

    if safety_mode == "conservative":
        adjusted["mission_time"] *= 1.03 + 0.01 * risk_index
        adjusted["verification_time"] *= 1.02 + 0.01 * risk_index
        adjusted["encirclement_time"] *= 1.04 + 0.01 * risk_index
        adjusted["formation_error"] *= max(0.72, 0.88 - 0.02 * risk_index)
        adjusted["rescue_success"] = min(0.999, adjusted["rescue_success"] * (1.03 + 0.015 * risk_index))
        adjusted["communication_load"] *= 1.06
    elif safety_mode == "balanced":
        adjusted["formation_error"] *= max(0.86, 0.96 - 0.01 * risk_index)
        adjusted["rescue_success"] = min(0.999, adjusted["rescue_success"] * (1.01 + 0.005 * risk_index))
    else:
        adjusted["mission_time"] *= 0.97
        adjusted["encirclement_time"] *= 0.97
        adjusted["formation_error"] *= 1.10 + 0.02 * risk_index
        adjusted["rescue_success"] *= max(0.90, 0.97 - 0.01 * risk_index)
        adjusted["communication_load"] *= 0.95

    adjusted["utility_score"] = adjusted["utility_score"] * (0.92 + 0.12 * adjusted["rescue_success"])
    adjusted["total_path_length"] *= 1.02 if safety_mode == "conservative" else 0.99
    adjusted["rescue_success"] = max(0.0, min(0.999, adjusted["rescue_success"]))
    return adjusted


def derive_violation_rate(metrics: Dict[str, float], scenario, hard_constraint_count: int) -> float:
    violation = (
        0.022 * metrics["formation_error"]
        + 0.028 * scenario.sea_state
        + 0.12 * (1.0 - scenario.communication_quality)
        - 0.16 * metrics["rescue_success"]
        + 0.008 * hard_constraint_count
    )
    return max(0.01, min(0.95, violation))


def compute_reward(metrics: Dict[str, float], reward_weights: Dict[str, float], violation_rate: float) -> float:
    detection_gain = 10.0 / (1.0 + metrics["detection_time"])
    verify_gain = 12.0 / (1.0 + metrics["verification_time"] - metrics["detection_time"])
    protect_gain = 22.0 * metrics["rescue_success"]
    utility_gain = 0.025 * metrics["utility_score"]
    safe_penalty = 2.2 * metrics["formation_error"] + 12.0 * violation_rate
    energy_penalty = metrics["total_path_length"] / 140.0
    mission_penalty = 0.18 * metrics["mission_time"]
    return (
        reward_weights["search"] * detection_gain
        + reward_weights["verify"] * verify_gain
        + reward_weights["protect"] * protect_gain
        + utility_gain
        - reward_weights["safe"] * safe_penalty
        - reward_weights["energy"] * energy_penalty
        - mission_penalty
    )


def compute_composite_score(
    metrics: Dict[str, float],
    violation_rate: float,
    weights: Dict[str, float] | None = None,
) -> float:
    active_weights = weights or COMPOSITE_SCORE_WEIGHTS
    return (
        active_weights["rescue_success"] * metrics["rescue_success"]
        - active_weights["violation_rate"] * violation_rate
        - active_weights["mission_time"] * metrics["mission_time"]
        - active_weights["formation_error"] * metrics["formation_error"]
        - active_weights["communication_load"] * metrics["communication_load"]
    )


def evaluate_bundle(evaluate_method, scenario, action: ActionBundle, reward_weights: Dict[str, float], hard_constraint_count: int, use_safety_projection: bool) -> Dict[str, float]:
    base_result = asdict(evaluate_method(scenario, action.method))
    metrics = dict(base_result)
    if use_safety_projection:
        metrics = apply_safety_projection(metrics, scenario, action.safety_mode)
    violation_rate = derive_violation_rate(metrics, scenario, hard_constraint_count)
    metrics["violation_rate"] = violation_rate
    metrics["reward"] = compute_reward(metrics, reward_weights, violation_rate)
    metrics["composite_score"] = compute_composite_score(metrics, violation_rate)
    metrics["action_label"] = action.label
    return metrics
