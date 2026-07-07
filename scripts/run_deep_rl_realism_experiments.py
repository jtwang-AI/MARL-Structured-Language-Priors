"""Deep-RL and higher-fidelity perturbation experiments for Chapter 5."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = ROOT / "simulator"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msar_sim.evaluation import evaluate_method  # noqa: E402
from msar_sim.scenario import generate_scenario  # noqa: E402

from env.chapter4_adapter import (  # noqa: E402
    ACTIONS,
    ActionBundle,
    compute_composite_score,
    compute_reward,
    derive_violation_rate,
    evaluate_bundle,
)
from planner.cognitive_adapter import CognitivePrior, infer_prior, load_schema  # noqa: E402
from policy.deep_actor_critic import (  # noqa: E402
    PPOActorCriticPolicy,
    evaluate_ppo_policy,
    train_ppo_actor_critic,
)
from policy.q_learning import TabularPolicy, evaluate_policy, train_policy  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
SCHEMA_PATH = ROOT / "planner" / "task_graph_schema.json"
Q_EPISODES = 420
DEEP_EPISODES = 1600
EVAL_COUNT = 96

REALISM_PROTOCOL = {
    "random_seed": 809,
    "hydrodynamic_load": {
        "distribution": "max(0, Normal(0.18 * sea_state + 0.12 * target_speed, 0.05))",
    },
    "sensor_error": {
        "distribution": "max(0, Normal(0.04 * sea_state + 0.10 * (1 - communication_quality), 0.015))",
    },
    "packet_drop_rate": {
        "distribution": "clip(Normal(0.55 * (1 - communication_quality) + 0.035 * sea_state, 0.035), 0, 0.85)",
    },
    "action_modifiers": {
        "conservative_safety": {"hydrodynamic_load": 0.88, "sensor_error": 0.84, "packet_drop_rate": 0.90},
        "behavior_distributed": {"packet_drop_rate": 0.92},
        "centralized_optimal": {"packet_drop_rate": 1.18},
    },
    "metric_transfer": {
        "mission_time": "x * (1 + 0.030 * hydrodynamic_load + 0.055 * packet_drop_rate)",
        "verification_time": "x * (1 + 0.025 * sensor_error + 0.030 * packet_drop_rate)",
        "encirclement_time": "x * (1 + 0.040 * hydrodynamic_load + 0.045 * packet_drop_rate)",
        "formation_error": "x * (1 + 0.070 * hydrodynamic_load + 0.110 * sensor_error)",
        "communication_load": "x * (1 + 0.32 * packet_drop_rate)",
        "total_path_length": "x * (1 + 0.020 * hydrodynamic_load)",
        "rescue_success": "x * max(0.70, 1 - 0.090 * packet_drop_rate - 0.060 * sensor_error)",
    },
}

METHOD_LABELS = {
    "fixed_rule": "Fixed rule",
    "q_full_stack": "Cognitive prior + Q-learning + safety",
    "deep_rl_only": "Deep actor-critic",
    "deep_prior_safety": "Cognitive prior + deep actor-critic + safety",
}


@dataclass(frozen=True)
class RealismParameters:
    hydrodynamic_load: float
    sensor_error: float
    packet_drop_rate: float


def neutral_prior(scenario) -> CognitivePrior:
    sea_bin = "highsea" if scenario.sea_state >= 3.5 else "lowsea"
    comm_bin = "weakcomm" if scenario.communication_quality <= 0.5 else "strongcomm"
    speed_bin = "fasttar" if scenario.target_speed >= 2.0 else "slowtar"
    return CognitivePrior(
        reward_weights={"search": 1.0, "verify": 1.0, "protect": 1.0, "safe": 1.0, "energy": 1.0},
        preferred_method="behavior_distributed",
        preferred_safety_mode="balanced",
        priority_focus="neutral",
        state_token=f"{scenario.regime}_{scenario.difficulty}_{sea_bin}_{comm_bin}_{speed_bin}_neutral",
    )


def build_prior(schema: dict, scenario, planner_enabled: bool) -> CognitivePrior:
    return infer_prior(schema, scenario) if planner_enabled else neutral_prior(scenario)


def scenario_sampler(index: int, rng: Random):
    regime = "stress" if rng.random() < 0.45 else "standard"
    seed = 22000 + 31 * index + rng.randint(0, 999)
    return generate_scenario(seed=seed, scenario_id=index, regime=regime)


def build_eval_sets(count: int = EVAL_COUNT):
    return {
        "standard": [generate_scenario(seed=31000 + idx, scenario_id=idx, regime="standard") for idx in range(count)],
        "stress": [generate_scenario(seed=33000 + idx, scenario_id=idx, regime="stress") for idx in range(count)],
    }


def one_hot(value: str, choices: Sequence[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in choices]


def state_features(scenario, prior: CognitivePrior, planner_enabled: bool) -> np.ndarray:
    reward_values = [prior.reward_weights[key] for key in ["search", "verify", "protect", "safe", "energy"]]
    reward_scale = max(sum(abs(item) for item in reward_values), 1e-6)
    normalized_rewards = [item / reward_scale for item in reward_values]
    values = [
        scenario.sea_state / 6.0,
        scenario.target_speed / 3.8,
        scenario.communication_quality,
        len([agent for agent in scenario.agents if agent.kind == "UAV"]) / 4.0,
        len([agent for agent in scenario.agents if agent.kind == "USV"]) / 8.0,
        1.0 if planner_enabled else 0.0,
    ]
    values.extend(one_hot(scenario.difficulty, ["easy", "medium", "hard"]))
    values.extend(one_hot(scenario.regime, ["standard", "stress"]))
    values.extend(normalized_rewards)
    values.extend(one_hot(prior.preferred_method, ["behavior_distributed", "homogeneous_distributed", "distance_only", "centralized_optimal"]))
    values.extend(one_hot(prior.preferred_safety_mode, ["balanced", "conservative"]))
    values.extend(one_hot(prior.priority_focus, ["search", "verify", "protect", "neutral"]))
    return np.asarray(values, dtype=np.float32)


def summarize_outcomes(name: str, regime: str, outcomes: Sequence[Dict[str, float]], extra: Dict[str, object] | None = None):
    row = {
        "method": name,
        "regime": regime,
        "mission_time": mean(item["mission_time"] for item in outcomes),
        "rescue_success": mean(item["rescue_success"] for item in outcomes),
        "formation_error": mean(item["formation_error"] for item in outcomes),
        "communication_load": mean(item["communication_load"] for item in outcomes),
        "violation_rate": mean(item["violation_rate"] for item in outcomes),
        "reward": mean(item["reward"] for item in outcomes),
        "composite_score": mean(item["composite_score"] for item in outcomes),
    }
    if extra:
        row.update(extra)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_evaluator(schema: dict):
    hard_count = len(schema["hard_constraints"])

    def evaluator(scenario, action, prior: CognitivePrior, safety_enabled: bool):
        return evaluate_bundle(evaluate_method, scenario, action, prior.reward_weights, hard_count, safety_enabled)

    return evaluator


def train_policies(schema: dict):
    evaluator = build_evaluator(schema)

    def prior_builder(scenario, enabled: bool):
        return build_prior(schema, scenario, enabled)

    q_policy = TabularPolicy(ACTIONS)
    q_trace = train_policy(
        q_policy,
        scenario_sampler,
        prior_builder,
        evaluator,
        Q_EPISODES,
        seed=101,
        planner_enabled=True,
        safety_enabled=True,
    )

    sample_scenario = generate_scenario(seed=30001, scenario_id=0, regime="standard")
    sample_prior = build_prior(schema, sample_scenario, True)
    state_dim = int(state_features(sample_scenario, sample_prior, True).shape[0])

    deep_rl = PPOActorCriticPolicy(ACTIONS, state_dim)
    deep_rl_trace = train_ppo_actor_critic(
        deep_rl,
        scenario_sampler,
        prior_builder,
        evaluator,
        state_features,
        episodes=DEEP_EPISODES,
        seed=103,
        planner_enabled=False,
        safety_enabled=False,
    )

    deep_prior_safety = PPOActorCriticPolicy(ACTIONS, state_dim)
    deep_prior_trace = train_ppo_actor_critic(
        deep_prior_safety,
        scenario_sampler,
        prior_builder,
        evaluator,
        state_features,
        episodes=DEEP_EPISODES,
        seed=107,
        planner_enabled=True,
        safety_enabled=True,
    )
    return {
        "q_full_stack": q_policy,
        "deep_rl_only": deep_rl,
        "deep_prior_safety": deep_prior_safety,
    }, {
        "q_full_stack": asdict(q_trace),
        "deep_rl_only": asdict(deep_rl_trace),
        "deep_prior_safety": asdict(deep_prior_trace),
    }


def evaluate_methods(schema: dict, policies: dict, eval_sets: Dict[str, Sequence]) -> List[Dict[str, object]]:
    evaluator = build_evaluator(schema)

    def prior_builder(scenario, enabled: bool):
        return build_prior(schema, scenario, enabled)

    rows: List[Dict[str, object]] = []
    for regime, scenarios in eval_sets.items():
        fixed = [
            evaluator(scenario, ActionBundle("behavior_distributed", "balanced"), neutral_prior(scenario), False)
            for scenario in scenarios
        ]
        q_full = evaluate_policy(policies["q_full_stack"], list(scenarios), prior_builder, evaluator, True, True)
        deep_only = evaluate_ppo_policy(
            policies["deep_rl_only"],
            scenarios,
            prior_builder,
            evaluator,
            state_features,
            planner_enabled=False,
            safety_enabled=False,
        )
        deep_prior = evaluate_ppo_policy(
            policies["deep_prior_safety"],
            scenarios,
            prior_builder,
            evaluator,
            state_features,
            planner_enabled=True,
            safety_enabled=True,
        )
        rows.extend(
            [
                summarize_outcomes("fixed_rule", regime, fixed),
                summarize_outcomes("q_full_stack", regime, q_full),
                summarize_outcomes("deep_rl_only", regime, deep_only),
                summarize_outcomes("deep_prior_safety", regime, deep_prior),
            ]
        )
    return rows


def recompute_after_realism(metrics: Dict[str, float], scenario, prior: CognitivePrior, hard_count: int) -> Dict[str, float]:
    adjusted = dict(metrics)
    adjusted["rescue_success"] = max(0.0, min(0.999, adjusted["rescue_success"]))
    violation_rate = derive_violation_rate(adjusted, scenario, hard_count)
    adjusted["violation_rate"] = violation_rate
    adjusted["reward"] = compute_reward(adjusted, prior.reward_weights, violation_rate)
    adjusted["composite_score"] = compute_composite_score(adjusted, violation_rate)
    return adjusted


def sample_realism_parameters(scenario, action: ActionBundle, rng: Random) -> RealismParameters:
    sea = scenario.sea_state
    hydrodynamic_load = max(0.0, rng.gauss(0.18 * sea + 0.12 * scenario.target_speed, 0.05))
    sensor_error = max(0.0, rng.gauss(0.04 * sea + 0.10 * (1.0 - scenario.communication_quality), 0.015))
    packet_drop_rate = max(0.0, min(0.85, rng.gauss(0.55 * (1.0 - scenario.communication_quality) + 0.035 * sea, 0.035)))

    if action.safety_mode == "conservative":
        hydrodynamic_load *= 0.88
        sensor_error *= 0.84
        packet_drop_rate *= 0.90
    if action.method == "behavior_distributed":
        packet_drop_rate *= 0.92
    elif action.method == "centralized_optimal":
        packet_drop_rate *= 1.18

    return RealismParameters(
        hydrodynamic_load=hydrodynamic_load,
        sensor_error=sensor_error,
        packet_drop_rate=packet_drop_rate,
    )


def apply_realism_layer(metrics: Dict[str, float], scenario, action: ActionBundle, prior: CognitivePrior, hard_count: int, rng: Random):
    adjusted = dict(metrics)
    params = sample_realism_parameters(scenario, action, rng)
    hydrodynamic_load = params.hydrodynamic_load
    sensor_error = params.sensor_error
    packet_drop = params.packet_drop_rate

    adjusted["mission_time"] *= 1.0 + 0.030 * hydrodynamic_load + 0.055 * packet_drop
    adjusted["verification_time"] *= 1.0 + 0.025 * sensor_error + 0.030 * packet_drop
    adjusted["encirclement_time"] *= 1.0 + 0.040 * hydrodynamic_load + 0.045 * packet_drop
    adjusted["formation_error"] *= 1.0 + 0.070 * hydrodynamic_load + 0.110 * sensor_error
    adjusted["communication_load"] *= 1.0 + 0.32 * packet_drop
    adjusted["total_path_length"] *= 1.0 + 0.020 * hydrodynamic_load
    adjusted["rescue_success"] *= max(0.70, 1.0 - 0.090 * packet_drop - 0.060 * sensor_error)
    adjusted = recompute_after_realism(adjusted, scenario, prior, hard_count)
    adjusted["hydrodynamic_load"] = hydrodynamic_load
    adjusted["sensor_error"] = sensor_error
    adjusted["packet_drop_rate"] = packet_drop
    return adjusted


def evaluate_with_realism(schema: dict, policies: dict, scenarios: Sequence) -> List[Dict[str, object]]:
    evaluator = build_evaluator(schema)
    hard_count = len(schema["hard_constraints"])
    rng = Random(809)
    rows: List[Dict[str, object]] = []

    def prior_builder(scenario, enabled: bool):
        return build_prior(schema, scenario, enabled)

    method_specs = [
        ("fixed_rule", None, False, False),
        ("q_full_stack", policies["q_full_stack"], True, True),
        ("deep_rl_only", policies["deep_rl_only"], False, False),
        ("deep_prior_safety", policies["deep_prior_safety"], True, True),
    ]
    for method_name, policy, planner_enabled, safety_enabled in method_specs:
        nominal: List[Dict[str, float]] = []
        realistic: List[Dict[str, float]] = []
        for scenario in scenarios:
            prior = prior_builder(scenario, planner_enabled)
            if method_name == "fixed_rule":
                action = ActionBundle("behavior_distributed", "balanced")
            elif method_name == "q_full_stack":
                action = policy.greedy_action(prior.state_token)
            else:
                action = policy.greedy_action(state_features(scenario, prior, planner_enabled))
            base = evaluator(scenario, action, prior, safety_enabled)
            nominal.append(base)
            realistic.append(apply_realism_layer(base, scenario, action, prior, hard_count, rng))
        rows.append(summarize_outcomes(method_name, "stress", nominal, {"layer": "nominal"}))
        rows.append(
            summarize_outcomes(
                method_name,
                "stress",
                realistic,
                {
                    "layer": "hydro_sensor_comm",
                    "hydrodynamic_load": mean(item["hydrodynamic_load"] for item in realistic),
                    "sensor_error": mean(item["sensor_error"] for item in realistic),
                    "packet_drop_rate": mean(item["packet_drop_rate"] for item in realistic),
                },
            )
        )
    return rows


def export_deep_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Deep high-level reinforcement-learning comparison.}",
        "  \\label{tab:deep_rl_comparison}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llrrrr}",
        "    \\toprule",
        "    Method & Regime & Rescue & Formation err. & Violation & Score \\\\",
        "    \\midrule",
    ]
    for regime in ["standard", "stress"]:
        for method in ["fixed_rule", "q_full_stack", "deep_rl_only", "deep_prior_safety"]:
            row = next(item for item in rows if item["method"] == method and item["regime"] == regime)
            lines.append(
                f"    {METHOD_LABELS[method]} & {regime.title()} & {row['rescue_success']:.3f} & {row['formation_error']:.2f} & {row['violation_rate']:.3f} & {row['composite_score']:.2f} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}}", "\\end{table}"])
    (RESULTS_DIR / "deep_rl_comparison_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_realism_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Stress-regime evaluation with hydrodynamic, sensor-noise, and packet-drop perturbations.}",
        "  \\label{tab:realism_layer}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llrrrrr}",
        "    \\toprule",
        "    Method & Layer & Rescue & Formation err. & Violation & Drop rate & Score \\\\",
        "    \\midrule",
    ]
    for method in ["fixed_rule", "q_full_stack", "deep_rl_only", "deep_prior_safety"]:
        for layer in ["nominal", "hydro_sensor_comm"]:
            row = next(item for item in rows if item["method"] == method and item["layer"] == layer)
            drop = row.get("packet_drop_rate", 0.0)
            layer_label = "Nominal" if layer == "nominal" else "Hydro+sensor+comm"
            lines.append(
                f"    {METHOD_LABELS[method]} & {layer_label} & {row['rescue_success']:.3f} & {row['formation_error']:.2f} & {row['violation_rate']:.3f} & {drop:.3f} & {row['composite_score']:.2f} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}}", "\\end{table}"])
    (RESULTS_DIR / "realism_layer_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_deep_results(rows: Sequence[Dict[str, object]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["fixed_rule", "q_full_stack", "deep_rl_only", "deep_prior_safety"]
    colors = ["#4c78a8", "#2a9d8f", "#6c5ce7", "#c84b39"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    stress = {row["method"]: row for row in rows if row["regime"] == "stress"}
    axes[0].bar(range(len(methods)), [stress[m]["composite_score"] for m in methods], color=colors)
    axes[1].bar(range(len(methods)), [stress[m]["violation_rate"] for m in methods], color=colors)
    for axis in axes:
        axis.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_title("Stress composite score")
    axes[0].set_ylabel("Score")
    axes[1].set_title("Stress violation rate")
    axes[1].set_ylabel("Violation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "deep_rl_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_realism_results(rows: Sequence[Dict[str, object]]) -> None:
    methods = ["fixed_rule", "q_full_stack", "deep_rl_only", "deep_prior_safety"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    width = 0.36
    x = np.arange(len(methods))
    lookup = {(row["method"], row["layer"]): row for row in rows}
    axes[0].bar(x - width / 2, [lookup[(m, "nominal")]["composite_score"] for m in methods], width=width, label="Nominal", color="#4c78a8")
    axes[0].bar(x + width / 2, [lookup[(m, "hydro_sensor_comm")]["composite_score"] for m in methods], width=width, label="Hydro+sensor+comm", color="#c84b39")
    axes[1].bar(x - width / 2, [lookup[(m, "nominal")]["violation_rate"] for m in methods], width=width, label="Nominal", color="#2a9d8f")
    axes[1].bar(x + width / 2, [lookup[(m, "hydro_sensor_comm")]["violation_rate"] for m in methods], width=width, label="Hydro+sensor+comm", color="#f4a261")
    for axis in axes:
        axis.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Violation")
    axes[0].set_title("Realism-layer score degradation")
    axes[1].set_title("Realism-layer safety degradation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "realism_layer_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_schema(SCHEMA_PATH)
    policies, traces = train_policies(schema)
    eval_sets = build_eval_sets()

    deep_rows = evaluate_methods(schema, policies, eval_sets)
    realism_rows = evaluate_with_realism(schema, policies, eval_sets["stress"])

    write_csv(RESULTS_DIR / "deep_rl_comparison.csv", deep_rows)
    write_csv(RESULTS_DIR / "realism_layer_results.csv", realism_rows)
    write_json(
        RESULTS_DIR / "deep_rl_realism_summary.json",
        {
            "deep_rl": deep_rows,
            "realism": realism_rows,
            "training": {
                name: {
                    "final_reward": trace["moving_rewards"][-1],
                    "final_score": trace["moving_scores"][-1],
                    "final_violation": trace["moving_violations"][-1],
                }
                for name, trace in traces.items()
            },
            "state_dim": int(len(state_features(generate_scenario(seed=1, scenario_id=0), build_prior(schema, generate_scenario(seed=1, scenario_id=0), True), True))),
            "deep_episodes": DEEP_EPISODES,
            "q_episodes": Q_EPISODES,
            "realism_protocol": REALISM_PROTOCOL,
        },
    )
    export_deep_table(deep_rows)
    export_realism_table(realism_rows)
    plot_deep_results(deep_rows)
    plot_realism_results(realism_rows)
    print("Deep-RL and realism experiments completed")
    print(f"- deep RL table: {RESULTS_DIR / 'deep_rl_comparison_table.tex'}")
    print(f"- realism table: {RESULTS_DIR / 'realism_layer_table.tex'}")
    print(f"- summary: {RESULTS_DIR / 'deep_rl_realism_summary.json'}")


if __name__ == "__main__":
    main()
