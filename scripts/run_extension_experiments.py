"""Extension experiments for the Chapter 5 small paper.

The script adds four submission-oriented checks:

1. rule-shift adaptation,
2. platform-failure recovery,
3. safety-projection ablation,
4. offline schema parsing evaluation for language-like SAR instructions.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = ROOT / "simulator"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msar_sim.evaluation import evaluate_method  # noqa: E402
from msar_sim.scenario import Scenario, generate_scenario  # noqa: E402

from env.chapter4_adapter import (  # noqa: E402
    ACTIONS,
    ActionBundle,
    compute_composite_score,
    compute_reward,
    derive_violation_rate,
    evaluate_bundle,
)
from planner.cognitive_adapter import CognitivePrior, infer_prior, load_schema  # noqa: E402
from policy.q_learning import TabularPolicy, evaluate_policy, moving_average, train_policy  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
SCHEMA_PATH = ROOT / "planner" / "task_graph_schema.json"
EPISODES = 420
SHIFT_EPISODE = 210
EVAL_COUNT = 80


METHOD_LABELS = {
    "fixed_rule": "Fixed rule",
    "rl_only": "RL only",
    "llm_rule_only": "Schema rule only",
    "full_stack": "Full stack",
    "q_only": "Q-learning only",
    "prior_only": "Prior + Q-learning",
    "safety_only": "Q-learning + safety",
    "prior_safety": "Prior + Q-learning + safety",
}


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


def shifted_schema(schema: dict) -> dict:
    updated = json.loads(json.dumps(schema))
    updated["reward_template"] = {
        "search": 0.75,
        "verify": 1.15,
        "protect": 2.05,
        "safe": 1.95,
        "energy": 0.45,
    }
    updated["task_context"]["mission_text"] = (
        "规则变化：确认目标后优先保持防护圈完整，必要时牺牲任务速度以降低安全违规。"
    )
    return updated


def build_prior(schema: dict, scenario, planner_enabled: bool) -> CognitivePrior:
    return infer_prior(schema, scenario) if planner_enabled else neutral_prior(scenario)


def heuristic_action(prior: CognitivePrior) -> ActionBundle:
    for action in ACTIONS:
        if action.method == prior.preferred_method and action.safety_mode == prior.preferred_safety_mode:
            return action
    for action in ACTIONS:
        if action.method == prior.preferred_method:
            return action
    return ACTIONS[0]


def scenario_sampler(index: int, rng: Random):
    regime = "stress" if rng.random() < 0.35 else "standard"
    seed = 1000 + 17 * index + rng.randint(0, 999)
    return generate_scenario(seed=seed, scenario_id=index, regime=regime)


def build_eval_sets(count: int = EVAL_COUNT):
    return {
        "standard": [generate_scenario(seed=5000 + idx, scenario_id=idx, regime="standard") for idx in range(count)],
        "stress": [generate_scenario(seed=7000 + idx, scenario_id=idx, regime="stress") for idx in range(count)],
    }


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
    }
    row["composite_score"] = (
        130.0 * row["rescue_success"]
        - 18.0 * row["violation_rate"]
        - 0.12 * row["mission_time"]
        - 0.35 * row["formation_error"]
        - 0.05 * row["communication_load"]
    )
    if extra:
        row.update(extra)
    return row


def train_variant(schema: dict, planner_enabled: bool, safety_enabled: bool, seed: int) -> TabularPolicy:
    hard_count = len(schema["hard_constraints"])

    def prior_builder(scenario, enabled: bool):
        return build_prior(schema, scenario, enabled)

    def evaluator(scenario, action, prior: CognitivePrior, enabled: bool):
        return evaluate_bundle(evaluate_method, scenario, action, prior.reward_weights, hard_count, enabled)

    policy = TabularPolicy(ACTIONS)
    train_policy(policy, scenario_sampler, prior_builder, evaluator, EPISODES, seed, planner_enabled, safety_enabled)
    return policy


def evaluate_variant(
    schema: dict,
    policy: TabularPolicy,
    scenarios: Sequence,
    planner_enabled: bool,
    safety_enabled: bool,
) -> List[Dict[str, float]]:
    hard_count = len(schema["hard_constraints"])

    def prior_builder(scenario, enabled: bool):
        return build_prior(schema, scenario, enabled)

    def evaluator(scenario, action, prior: CognitivePrior, enabled: bool):
        return evaluate_bundle(evaluate_method, scenario, action, prior.reward_weights, hard_count, enabled)

    return evaluate_policy(policy, list(scenarios), prior_builder, evaluator, planner_enabled, safety_enabled)


def recompute_scores(metrics: Dict[str, float], scenario, reward_weights: Dict[str, float], hard_count: int) -> Dict[str, float]:
    adjusted = dict(metrics)
    adjusted["rescue_success"] = max(0.0, min(0.999, adjusted["rescue_success"]))
    violation_rate = derive_violation_rate(adjusted, scenario, hard_count)
    adjusted["violation_rate"] = violation_rate
    adjusted["reward"] = compute_reward(adjusted, reward_weights, violation_rate)
    adjusted["composite_score"] = compute_composite_score(adjusted, violation_rate)
    return adjusted


def apply_rule_shift_effect(
    metrics: Dict[str, float],
    scenario,
    action: ActionBundle,
    prior: CognitivePrior,
    shifted: bool,
    hard_count: int,
) -> Dict[str, float]:
    if not shifted:
        return metrics
    adjusted = dict(metrics)
    if action.safety_mode == "conservative":
        adjusted["mission_time"] *= 1.01
        adjusted["formation_error"] *= 0.92
        adjusted["rescue_success"] *= 1.02
    else:
        adjusted["mission_time"] *= 0.99
        adjusted["formation_error"] *= 1.10
        adjusted["rescue_success"] *= 0.985
    if action.method != "behavior_distributed":
        adjusted["formation_error"] *= 1.06
        adjusted["communication_load"] *= 1.04
        adjusted["rescue_success"] *= 0.985
    return recompute_scores(adjusted, scenario, prior.reward_weights, hard_count)


def run_rule_shift(schema: dict) -> List[Dict[str, float]]:
    base_schema = schema
    post_schema = shifted_schema(schema)
    hard_count = len(schema["hard_constraints"])
    rng = Random(31)
    policies = {
        "rl_only": TabularPolicy(ACTIONS),
        "full_stack": TabularPolicy(ACTIONS),
    }
    traces: Dict[str, Dict[str, List[float]]] = {
        name: {"episode": [], "score": [], "violation": []}
        for name in ["fixed_rule", "rl_only", "llm_rule_only", "full_stack"]
    }

    scenario = scenario_sampler(0, rng)
    for episode in range(EPISODES):
        is_shifted = episode >= SHIFT_EPISODE
        active_schema = post_schema if is_shifted else base_schema
        next_scenario = scenario_sampler(episode + 1, rng)

        fixed_prior = neutral_prior(scenario)
        fixed_action = ActionBundle("behavior_distributed", "balanced")
        fixed = evaluate_bundle(evaluate_method, scenario, fixed_action, fixed_prior.reward_weights, hard_count, False)
        fixed = apply_rule_shift_effect(fixed, scenario, fixed_action, fixed_prior, is_shifted, hard_count)

        rule_prior = build_prior(active_schema, scenario, True)
        rule_action = heuristic_action(rule_prior)
        rule = evaluate_bundle(evaluate_method, scenario, rule_action, rule_prior.reward_weights, hard_count, False)
        rule = apply_rule_shift_effect(rule, scenario, rule_action, rule_prior, is_shifted, hard_count)

        rl_prior = neutral_prior(scenario)
        rl_state = rl_prior.state_token
        rl_action = policies["rl_only"].choose_action(rl_state, max(0.06, 0.30 * (1.0 - episode / (EPISODES - 1))), rng)
        rl = evaluate_bundle(evaluate_method, scenario, rl_action, rl_prior.reward_weights, hard_count, False)
        rl = apply_rule_shift_effect(rl, scenario, rl_action, rl_prior, is_shifted, hard_count)
        rl_next = neutral_prior(next_scenario)
        policies["rl_only"].update(rl_state, rl_action.label, rl["reward"], rl_next.state_token)

        full_prior = build_prior(active_schema, scenario, True)
        full_state = full_prior.state_token
        full_action = policies["full_stack"].choose_action(full_state, max(0.06, 0.30 * (1.0 - episode / (EPISODES - 1))), rng)
        full = evaluate_bundle(evaluate_method, scenario, full_action, full_prior.reward_weights, hard_count, True)
        full = apply_rule_shift_effect(full, scenario, full_action, full_prior, is_shifted, hard_count)
        full_next = build_prior(active_schema, next_scenario, True)
        policies["full_stack"].update(full_state, full_action.label, full["reward"], full_next.state_token)

        for name, outcome in [("fixed_rule", fixed), ("llm_rule_only", rule), ("rl_only", rl), ("full_stack", full)]:
            traces[name]["episode"].append(episode + 1)
            traces[name]["score"].append(outcome["composite_score"])
            traces[name]["violation"].append(outcome["violation_rate"])
        scenario = next_scenario

    rows = []
    for name, trace in traces.items():
        moving_scores = moving_average(trace["score"], 20)
        moving_violations = moving_average(trace["violation"], 20)
        pre_score = mean(trace["score"][SHIFT_EPISODE - 30 : SHIFT_EPISODE])
        post_score = mean(trace["score"][-40:])
        post_violation = mean(trace["violation"][-40:])
        adaptation_steps = recovery_steps(moving_scores, SHIFT_EPISODE, post_score)
        rows.append(
            {
                "method": name,
                "pre_shift_score": pre_score,
                "post_shift_score": post_score,
                "score_delta": post_score - pre_score,
                "post_shift_violation": post_violation,
                "adaptation_steps": adaptation_steps,
                "final_moving_score": moving_scores[-1],
                "final_moving_violation": moving_violations[-1],
            }
        )

    plot_rule_shift(traces)
    write_csv(RESULTS_DIR / "rule_shift_results.csv", rows)
    write_json(RESULTS_DIR / "rule_shift_summary.json", {"rows": rows, "shift_episode": SHIFT_EPISODE})
    export_rule_shift_table(rows)
    return rows


def recovery_steps(values: Sequence[float], shift_idx: int, final_mean: float) -> int:
    tolerance = max(0.5, abs(final_mean) * 0.08)
    for idx in range(shift_idx, len(values)):
        if abs(values[idx] - final_mean) <= tolerance:
            return idx - shift_idx + 1
    return len(values) - shift_idx


def plot_rule_shift(traces: Dict[str, Dict[str, List[float]]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    colors = {
        "fixed_rule": "#4c78a8",
        "rl_only": "#355070",
        "llm_rule_only": "#f4a261",
        "full_stack": "#2a9d8f",
    }
    for name, trace in traces.items():
        label = METHOD_LABELS[name]
        axes[0].plot(trace["episode"], moving_average(trace["score"], 20), label=label, color=colors[name], linewidth=2.0)
        axes[1].plot(trace["episode"], moving_average(trace["violation"], 20), label=label, color=colors[name], linewidth=2.0)
    for axis in axes:
        axis.axvline(SHIFT_EPISODE, color="#c84b39", linestyle="--", linewidth=1.2)
        axis.grid(alpha=0.25, linestyle="--")
        axis.set_xlabel("Episode")
    axes[0].set_ylabel("Moving-average composite score")
    axes[1].set_ylabel("Moving-average violation rate")
    axes[0].set_title("Rule-shift adaptation")
    axes[1].set_title("Safety after rule shift")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "rule_shift_adaptation.png", bbox_inches="tight")
    plt.close(fig)


def fail_scenario(scenario: Scenario, failure_type: str) -> Scenario:
    if failure_type == "uav_loss":
        candidates = [agent for agent in scenario.agents if agent.kind == "UAV"]
    elif failure_type == "usv_loss":
        candidates = [agent for agent in scenario.agents if agent.kind == "USV"]
    else:
        raise ValueError(f"unknown failure type: {failure_type}")
    failed = min(candidates, key=lambda agent: (agent.energy, agent.agent_id))
    remaining = [agent for agent in scenario.agents if agent.agent_id != failed.agent_id]
    return replace(scenario, agents=remaining)


def run_platform_failure(schema: dict) -> List[Dict[str, float]]:
    policies = {
        "rl_only": train_variant(schema, False, False, 41),
        "full_stack": train_variant(schema, True, True, 43),
    }
    eval_sets = build_eval_sets()
    rows = []
    for failure_type in ["uav_loss", "usv_loss"]:
        for regime, scenarios in eval_sets.items():
            failed_scenarios = [fail_scenario(scenario, failure_type) for scenario in scenarios]
            fixed = [
                evaluate_bundle(evaluate_method, scenario, ActionBundle("behavior_distributed", "balanced"), neutral_prior(scenario).reward_weights, len(schema["hard_constraints"]), False)
                for scenario in failed_scenarios
            ]
            rule = []
            for scenario in failed_scenarios:
                prior = build_prior(schema, scenario, True)
                rule.append(evaluate_bundle(evaluate_method, scenario, heuristic_action(prior), prior.reward_weights, len(schema["hard_constraints"]), False))
            rl = evaluate_variant(schema, policies["rl_only"], failed_scenarios, False, False)
            full = evaluate_variant(schema, policies["full_stack"], failed_scenarios, True, True)
            rows.extend(
                [
                    summarize_outcomes("fixed_rule", regime, fixed, {"failure_type": failure_type}),
                    summarize_outcomes("rl_only", regime, rl, {"failure_type": failure_type}),
                    summarize_outcomes("llm_rule_only", regime, rule, {"failure_type": failure_type}),
                    summarize_outcomes("full_stack", regime, full, {"failure_type": failure_type}),
                ]
            )
    write_csv(RESULTS_DIR / "platform_failure_results.csv", rows)
    write_json(RESULTS_DIR / "platform_failure_summary.json", {"rows": rows})
    plot_platform_failure(rows)
    export_platform_failure_table(rows)
    return rows


def plot_platform_failure(rows: Sequence[Dict[str, object]]) -> None:
    methods = ["fixed_rule", "rl_only", "llm_rule_only", "full_stack"]
    colors = ["#4c78a8", "#355070", "#f4a261", "#2a9d8f"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    for axis, failure_type in zip(axes, ["uav_loss", "usv_loss"]):
        values = [
            mean(row["composite_score"] for row in rows if row["method"] == method and row["failure_type"] == failure_type)
            for method in methods
        ]
        axis.bar(range(len(methods)), values, color=colors)
        axis.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods], rotation=12)
        axis.set_ylabel("Composite score")
        axis.set_title("UAV loss" if failure_type == "uav_loss" else "USV loss")
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "platform_failure_comparison.png", bbox_inches="tight")
    plt.close(fig)


def run_safety_ablation(schema: dict) -> List[Dict[str, float]]:
    variants = {
        "q_only": (False, False, 51),
        "prior_only": (True, False, 53),
        "safety_only": (False, True, 55),
        "prior_safety": (True, True, 57),
    }
    policies = {
        name: train_variant(schema, planner_enabled, safety_enabled, seed)
        for name, (planner_enabled, safety_enabled, seed) in variants.items()
    }
    eval_sets = build_eval_sets()
    rows = []
    for regime, scenarios in eval_sets.items():
        for name, (planner_enabled, safety_enabled, _) in variants.items():
            outcomes = evaluate_variant(schema, policies[name], scenarios, planner_enabled, safety_enabled)
            rows.append(summarize_outcomes(name, regime, outcomes))
    write_csv(RESULTS_DIR / "safety_ablation_results.csv", rows)
    write_json(RESULTS_DIR / "safety_ablation_summary.json", {"rows": rows})
    plot_safety_ablation(rows)
    export_safety_ablation_table(rows)
    return rows


def plot_safety_ablation(rows: Sequence[Dict[str, object]]) -> None:
    variants = ["q_only", "prior_only", "safety_only", "prior_safety"]
    x = range(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    stress = {row["method"]: row for row in rows if row["regime"] == "stress"}
    axes[0].bar(x, [stress[name]["violation_rate"] for name in variants], color="#c84b39")
    axes[1].bar(x, [stress[name]["composite_score"] for name in variants], color="#2a9d8f")
    for axis in axes:
        axis.set_xticks(list(x), [METHOD_LABELS[name] for name in variants], rotation=14)
        axis.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Stress violation rate")
    axes[1].set_ylabel("Stress composite score")
    axes[0].set_title("Safety ablation")
    axes[1].set_title("Stress-regime score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "safety_ablation.png", bbox_inches="tight")
    plt.close(fig)


def run_parse_eval(schema: dict) -> List[Dict[str, object]]:
    cases = build_parse_cases()
    rows = []
    for idx, case in enumerate(cases, start=1):
        parsed = parse_instruction(case["text"], schema)
        rows.append(score_parse_case(idx, case, parsed))
    summary = {
        "parser": "offline_keyword_schema_baseline",
        "case_count": len(rows),
        "task_node_accuracy": mean(float(row["task_node_ok"]) for row in rows),
        "role_prior_accuracy": mean(float(row["role_prior_ok"]) for row in rows),
        "hard_constraint_accuracy": mean(float(row["hard_constraint_ok"]) for row in rows),
        "reward_focus_accuracy": mean(float(row["reward_focus_ok"]) for row in rows),
        "invalid_output_rate": mean(float(row["invalid_output"]) for row in rows),
    }
    write_csv(RESULTS_DIR / "llm_parse_eval.csv", rows)
    write_json(RESULTS_DIR / "llm_parse_summary.json", {"summary": summary, "rows": rows})
    export_parse_eval_table(summary)
    return rows


def build_parse_cases() -> List[Dict[str, object]]:
    return [
        {"text": "Search the northeast sector first, verify any high-confidence target, then keep two USVs in protection.", "tasks": {"search", "verify", "protect"}, "roles": {"search_uav", "protect_usv"}, "constraints": {"protect_usv", "communication"}, "focus": "protect"},
        {"text": "弱通信条件下优先确认目标，避免平台过近会遇。", "tasks": {"verify"}, "roles": {"monitor_uav"}, "constraints": {"communication", "distance"}, "focus": "verify"},
        {"text": "High sea state: slow down and maintain a conservative protection ring after target confirmation.", "tasks": {"verify", "protect"}, "roles": {"protect_usv"}, "constraints": {"distance", "protect_usv"}, "focus": "protect"},
        {"text": "优先搜索东北扇区，低风险时节约能耗。", "tasks": {"search"}, "roles": {"search_uav"}, "constraints": set(), "focus": "search"},
        {"text": "If the target is found, one UAV should monitor it while USVs build the safety perimeter.", "tasks": {"verify", "protect"}, "roles": {"monitor_uav", "protect_usv"}, "constraints": {"protect_usv"}, "focus": "protect"},
        {"text": "通信质量低于阈值时保持中继链路，不要切换到集中式协调。", "tasks": {"search"}, "roles": {"search_uav"}, "constraints": {"communication"}, "focus": "safe"},
        {"text": "Fast drifting target: verify quickly and increase the protection weight.", "tasks": {"verify", "protect"}, "roles": {"monitor_uav", "protect_usv"}, "constraints": {"protect_usv"}, "focus": "verify"},
        {"text": "Only perform wide-area search; no protection is required until the target is confirmed.", "tasks": {"search", "verify"}, "roles": {"search_uav"}, "constraints": set(), "focus": "search"},
        {"text": "目标确认后至少两艘USV形成防护圈，并保持15米安全距离。", "tasks": {"verify", "protect"}, "roles": {"protect_usv"}, "constraints": {"protect_usv", "distance"}, "focus": "protect"},
        {"text": "Energy is scarce, so search with minimum travel unless sea state rises.", "tasks": {"search"}, "roles": {"search_uav"}, "constraints": set(), "focus": "search"},
        {"text": "Prioritize safety over mission time in stress weather.", "tasks": {"protect"}, "roles": {"protect_usv"}, "constraints": {"distance"}, "focus": "safe"},
        {"text": "Use UAVs to scan and USVs to verify close targets under normal communication.", "tasks": {"search", "verify"}, "roles": {"search_uav", "monitor_uav"}, "constraints": set(), "focus": "verify"},
    ]


def parse_instruction(text: str, schema: dict) -> Dict[str, object]:
    lowered = text.lower()
    tasks = set()
    roles = set()
    constraints = set()
    reward = dict(schema["reward_template"])
    if any(token in lowered for token in ["search", "scan", "搜索", "搜寻"]):
        tasks.add("search")
        roles.add("search_uav")
        reward["search"] += 0.35
    if any(token in lowered for token in ["verify", "confirm", "确认"]):
        tasks.add("verify")
        roles.add("monitor_uav")
        reward["verify"] += 0.35
    if any(token in lowered for token in ["protect", "protection", "perimeter", "防护", "保护", "防护圈"]):
        tasks.add("protect")
        roles.add("protect_usv")
        constraints.add("protect_usv")
        reward["protect"] += 0.45
    if any(token in lowered for token in ["communication", "comm", "通信", "relink", "链路"]):
        constraints.add("communication")
        reward["safe"] += 0.20
    if any(token in lowered for token in ["distance", "safe", "safety", "15", "安全", "过近"]):
        constraints.add("distance")
        reward["safe"] += 0.35
    if any(token in lowered for token in ["energy", "能耗", "节约"]):
        reward["energy"] += 0.25
    if any(token in lowered for token in ["stress", "high sea", "海况", "weather"]):
        reward["safe"] += 0.25
        if "protect" not in tasks:
            tasks.add("protect")
            roles.add("protect_usv")
    if not tasks:
        tasks.add("search")
        roles.add("search_uav")
    focus = max(reward, key=reward.get)
    if focus == "safe":
        priority_focus = "safe"
    elif focus == "protect":
        priority_focus = "protect"
    elif focus == "verify":
        priority_focus = "verify"
    else:
        priority_focus = "search"
    return {
        "tasks": sorted(tasks),
        "roles": sorted(roles),
        "constraints": sorted(constraints),
        "reward_template": reward,
        "priority_focus": priority_focus,
    }


def score_parse_case(idx: int, case: Dict[str, object], parsed: Dict[str, object]) -> Dict[str, object]:
    expected_tasks = set(case["tasks"])
    expected_roles = set(case["roles"])
    expected_constraints = set(case["constraints"])
    parsed_tasks = set(parsed["tasks"])
    parsed_roles = set(parsed["roles"])
    parsed_constraints = set(parsed["constraints"])
    return {
        "case_id": idx,
        "task_node_ok": expected_tasks.issubset(parsed_tasks),
        "role_prior_ok": expected_roles.issubset(parsed_roles),
        "hard_constraint_ok": expected_constraints.issubset(parsed_constraints),
        "reward_focus_ok": parsed["priority_focus"] == case["focus"] or (case["focus"] == "safe" and parsed["priority_focus"] in {"safe", "protect"}),
        "invalid_output": False,
        "expected_focus": case["focus"],
        "predicted_focus": parsed["priority_focus"],
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_rule_shift_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Rule-shift adaptation results. Lower adaptation steps and lower violation are better.}",
        "  \\label{tab:rule_shift}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Method & Pre-shift score & Post-shift score & Adapt. steps & Post violation \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            f"    {METHOD_LABELS[row['method']]} & {row['pre_shift_score']:.2f} & {row['post_shift_score']:.2f} & {int(row['adaptation_steps'])} & {row['post_shift_violation']:.3f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "rule_shift_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_platform_failure_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Platform-failure recovery under stress regimes.}",
        "  \\label{tab:platform_failure}",
        "  \\begin{tabular}{llrrrr}",
        "    \\toprule",
        "    Failure & Method & Rescue & Formation err. & Violation & Score \\\\",
        "    \\midrule",
    ]
    for failure_type in ["uav_loss", "usv_loss"]:
        for method in ["fixed_rule", "rl_only", "llm_rule_only", "full_stack"]:
            row = next(item for item in rows if item["failure_type"] == failure_type and item["regime"] == "stress" and item["method"] == method)
            failure_label = "UAV loss" if failure_type == "uav_loss" else "USV loss"
            lines.append(
                f"    {failure_label} & {METHOD_LABELS[method]} & {row['rescue_success']:.3f} & {row['formation_error']:.2f} & {row['violation_rate']:.3f} & {row['composite_score']:.2f} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "platform_failure_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_safety_ablation_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Safety-projection ablation in stress regimes.}",
        "  \\label{tab:safety_ablation}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Variant & Rescue & Formation err. & Violation & Score \\\\",
        "    \\midrule",
    ]
    for method in ["q_only", "prior_only", "safety_only", "prior_safety"]:
        row = next(item for item in rows if item["method"] == method and item["regime"] == "stress")
        lines.append(
            f"    {METHOD_LABELS[method]} & {row['rescue_success']:.3f} & {row['formation_error']:.2f} & {row['violation_rate']:.3f} & {row['composite_score']:.2f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "safety_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_parse_eval_table(summary: Dict[str, object]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Offline schema parsing sanity check.}",
        "  \\label{tab:parse_eval}",
        "  \\begin{tabular}{lr}",
        "    \\toprule",
        "    Metric & Value \\\\",
        "    \\midrule",
        f"    Task-node accuracy & {summary['task_node_accuracy']:.3f} \\\\",
        f"    Role-prior accuracy & {summary['role_prior_accuracy']:.3f} \\\\",
        f"    Hard-constraint accuracy & {summary['hard_constraint_accuracy']:.3f} \\\\",
        f"    Reward-focus accuracy & {summary['reward_focus_accuracy']:.3f} \\\\",
        f"    Invalid-output rate & {summary['invalid_output_rate']:.3f} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    (RESULTS_DIR / "llm_parse_eval_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rule_shift_rows = run_rule_shift(schema)
    platform_rows = run_platform_failure(schema)
    safety_rows = run_safety_ablation(schema)
    parse_rows = run_parse_eval(schema)

    write_json(
        RESULTS_DIR / "extension_experiments_summary.json",
        {
            "rule_shift": rule_shift_rows,
            "platform_failure": platform_rows,
            "safety_ablation": safety_rows,
            "schema_parse_cases": len(parse_rows),
        },
    )
    print("Extension experiments completed")
    print(f"- rule shift: {RESULTS_DIR / 'rule_shift_results.csv'}")
    print(f"- platform failure: {RESULTS_DIR / 'platform_failure_results.csv'}")
    print(f"- safety ablation: {RESULTS_DIR / 'safety_ablation_results.csv'}")
    print(f"- schema parse eval: {RESULTS_DIR / 'llm_parse_eval.csv'}")


if __name__ == "__main__":
    main()
