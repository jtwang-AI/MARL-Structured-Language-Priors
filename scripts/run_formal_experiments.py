"""Formal Chapter 5 experiments on top of the Chapter 4 simulator."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from random import Random
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = ROOT / "simulator"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msar_sim.evaluation import evaluate_method  # noqa: E402
from msar_sim.scenario import generate_scenario  # noqa: E402

from env.chapter4_adapter import ACTIONS, ActionBundle, evaluate_bundle  # noqa: E402
from planner.cognitive_adapter import CognitivePrior, infer_prior, load_schema  # noqa: E402
from policy.q_learning import TabularPolicy, evaluate_policy, train_policy  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
SCHEMA_PATH = ROOT / "planner" / "task_graph_schema.json"
TRAIN_FIGURE = FIGURE_DIR / "formal_training_curves.png"
REGIME_FIGURE = FIGURE_DIR / "formal_regime_comparison.png"
TABLE_PATH = RESULTS_DIR / "formal_results_table.tex"
CSV_PATH = RESULTS_DIR / "formal_results.csv"
JSON_PATH = RESULTS_DIR / "formal_summary.json"


def neutral_prior(scenario) -> CognitivePrior:
    reward_weights = {"search": 1.0, "verify": 1.0, "protect": 1.0, "safe": 1.0, "energy": 1.0}
    sea_bin = "highsea" if scenario.sea_state >= 3.5 else "lowsea"
    comm_bin = "weakcomm" if scenario.communication_quality <= 0.5 else "strongcomm"
    speed_bin = "fasttar" if scenario.target_speed >= 2.0 else "slowtar"
    return CognitivePrior(
        reward_weights=reward_weights,
        preferred_method="behavior_distributed",
        preferred_safety_mode="balanced",
        priority_focus="neutral",
        state_token=f"{scenario.regime}_{scenario.difficulty}_{sea_bin}_{comm_bin}_{speed_bin}_neutral",
    )


def scenario_sampler(index: int, rng: Random):
    regime = "stress" if rng.random() < 0.35 else "standard"
    seed = 1000 + 17 * index + rng.randint(0, 999)
    return generate_scenario(seed=seed, scenario_id=index, regime=regime)


def build_prior(schema, scenario, planner_enabled: bool) -> CognitivePrior:
    return infer_prior(schema, scenario) if planner_enabled else neutral_prior(scenario)


def evaluate_action(schema, scenario, action: ActionBundle, planner_enabled: bool, safety_enabled: bool):
    prior = build_prior(schema, scenario, planner_enabled)
    return evaluate_bundle(
        evaluate_method=evaluate_method,
        scenario=scenario,
        action=action,
        reward_weights=prior.reward_weights,
        hard_constraint_count=len(schema["hard_constraints"]),
        use_safety_projection=safety_enabled,
    )


def heuristic_action(prior: CognitivePrior) -> ActionBundle:
    for action in ACTIONS:
        if action.method == prior.preferred_method and action.safety_mode == prior.preferred_safety_mode:
            return action
    for action in ACTIONS:
        if action.method == prior.preferred_method:
            return action
    return ACTIONS[0]


def build_eval_sets():
    standard = [generate_scenario(seed=5000 + idx, scenario_id=idx, regime="standard") for idx in range(80)]
    stress = [generate_scenario(seed=7000 + idx, scenario_id=idx, regime="stress") for idx in range(80)]
    return {"standard": standard, "stress": stress}


def summarize_outcomes(name: str, regime: str, outcomes):
    mission_time = mean(item["mission_time"] for item in outcomes)
    rescue_success = mean(item["rescue_success"] for item in outcomes)
    formation_error = mean(item["formation_error"] for item in outcomes)
    communication_load = mean(item["communication_load"] for item in outcomes)
    violation_rate = mean(item["violation_rate"] for item in outcomes)
    reward = mean(item["reward"] for item in outcomes)
    composite_score = 130.0 * rescue_success - 18.0 * violation_rate - 0.12 * mission_time - 0.35 * formation_error - 0.05 * communication_load
    return {
        "method": name,
        "regime": regime,
        "mission_time": mission_time,
        "rescue_success": rescue_success,
        "formation_error": formation_error,
        "communication_load": communication_load,
        "violation_rate": violation_rate,
        "reward": reward,
        "composite_score": composite_score,
    }


def export_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "regime",
                "mission_time",
                "rescue_success",
                "formation_error",
                "communication_load",
                "violation_rate",
                "reward",
                "composite_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_json(payload):
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_table(rows):
    method_order = ["fixed_rule", "rl_only", "llm_rule_only", "full_stack"]
    labels = {
        "fixed_rule": "Fixed rule + auction",
        "rl_only": "RL only",
        "llm_rule_only": "LLM rule only",
        "full_stack": "Cognitive prior + Q-learning + safety projection",
    }
    regime_order = ["standard", "stress"]
    lookup = {(row["method"], row["regime"]): row for row in rows}
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Formal evaluation of high-level coordination strategies}",
        "  \\label{tab:formal_results}",
        "  \\begin{tabular}{p{4.0cm} p{1.6cm} p{1.6cm} p{1.6cm} p{1.6cm} p{1.6cm}}",
        "    \\toprule",
        "    Method & Regime & Mission time & Rescue success & Violation rate & Score \\\\",
        "    \\midrule",
    ]
    for method in method_order:
        for regime in regime_order:
            row = lookup[(method, regime)]
            lines.append(
                "    {label} & {regime} & {mission_time:.2f} & {rescue_success:.3f} & {violation_rate:.3f} & {reward:.2f} \\\\".format(
                    label=labels[method],
                    regime="Standard" if regime == "standard" else "Stress",
                    mission_time=row["mission_time"],
                    rescue_success=row["rescue_success"],
                    violation_rate=row["violation_rate"],
                    reward=row["composite_score"],
                )
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}", "\\end{table}"])
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_training(full_trace, rl_trace):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    axes[0].plot(full_trace.episodes, full_trace.moving_scores, label="Cognitive prior + Q-learning", linewidth=2.2, color="#2a9d8f")
    axes[0].plot(rl_trace.episodes, rl_trace.moving_scores, label="RL only", linewidth=2.0, color="#355070")
    axes[0].set_title("Moving-average composite score")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Moving-average score")
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(full_trace.episodes, full_trace.moving_violations, label="Cognitive prior + Q-learning", linewidth=2.2, color="#2a9d8f")
    axes[1].plot(rl_trace.episodes, rl_trace.moving_violations, label="RL only", linewidth=2.0, color="#355070")
    axes[1].set_title("Moving-average violation rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Moving-average violation")
    axes[1].grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(TRAIN_FIGURE, bbox_inches="tight")
    plt.close(fig)


def plot_regime_comparison(rows):
    labels = {
        "fixed_rule": "Fixed rule",
        "rl_only": "RL only",
        "llm_rule_only": "LLM rule only",
        "full_stack": "Full stack",
    }
    methods = ["fixed_rule", "rl_only", "llm_rule_only", "full_stack"]
    lookup = {(row["method"], row["regime"]): row for row in rows}
    x = list(range(len(methods)))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)

    standard_time = [lookup[(method, "standard")]["mission_time"] for method in methods]
    stress_time = [lookup[(method, "stress")]["mission_time"] for method in methods]
    standard_success = [lookup[(method, "standard")]["rescue_success"] for method in methods]
    stress_success = [lookup[(method, "stress")]["rescue_success"] for method in methods]

    axes[0].bar([item - width / 2 for item in x], standard_time, width=width, label="Standard", color="#4c78a8")
    axes[0].bar([item + width / 2 for item in x], stress_time, width=width, label="Stress", color="#e76f51")
    axes[0].set_xticks(x, [labels[method] for method in methods], rotation=12)
    axes[0].set_ylabel("Mission time")
    axes[0].set_title("Mission time across regimes")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar([item - width / 2 for item in x], standard_success, width=width, label="Standard", color="#2a9d8f")
    axes[1].bar([item + width / 2 for item in x], stress_success, width=width, label="Stress", color="#f4a261")
    axes[1].set_xticks(x, [labels[method] for method in methods], rotation=12)
    axes[1].set_ylabel("Rescue success")
    axes[1].set_title("Rescue success across regimes")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    handles, labels_plot = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_plot, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(REGIME_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    schema = load_schema(SCHEMA_PATH)
    hard_constraint_count = len(schema["hard_constraints"])

    def prior_builder(scenario, planner_enabled: bool) -> CognitivePrior:
        return build_prior(schema, scenario, planner_enabled)

    def evaluator(scenario, action, prior: CognitivePrior, safety_enabled: bool):
        return evaluate_bundle(
            evaluate_method=evaluate_method,
            scenario=scenario,
            action=action,
            reward_weights=prior.reward_weights,
            hard_constraint_count=hard_constraint_count,
            use_safety_projection=safety_enabled,
        )

    policy_full = TabularPolicy(ACTIONS)
    policy_rl = TabularPolicy(ACTIONS)
    full_trace = train_policy(policy_full, scenario_sampler, prior_builder, evaluator, episodes=420, seed=11, planner_enabled=True, safety_enabled=True)
    rl_trace = train_policy(policy_rl, scenario_sampler, prior_builder, evaluator, episodes=420, seed=23, planner_enabled=False, safety_enabled=False)

    eval_sets = build_eval_sets()
    summary_rows = []
    detailed_payload = {}

    for regime, scenarios in eval_sets.items():
        fixed_outcomes = [
            evaluator(scenario, ActionBundle("behavior_distributed", "balanced"), prior_builder(scenario, False), False)
            for scenario in scenarios
        ]
        rule_outcomes = []
        for scenario in scenarios:
            prior = prior_builder(scenario, True)
            rule_outcomes.append(evaluator(scenario, heuristic_action(prior), prior, False))

        rl_outcomes = evaluate_policy(policy_rl, scenarios, prior_builder, evaluator, planner_enabled=False, safety_enabled=False)
        full_outcomes = evaluate_policy(policy_full, scenarios, prior_builder, evaluator, planner_enabled=True, safety_enabled=True)

        regime_rows = [
            summarize_outcomes("fixed_rule", regime, fixed_outcomes),
            summarize_outcomes("rl_only", regime, rl_outcomes),
            summarize_outcomes("llm_rule_only", regime, rule_outcomes),
            summarize_outcomes("full_stack", regime, full_outcomes),
        ]
        summary_rows.extend(regime_rows)
        detailed_payload[regime] = {
            "fixed_rule": regime_rows[0],
            "rl_only": regime_rows[1],
            "llm_rule_only": regime_rows[2],
            "full_stack": regime_rows[3],
        }

    export_csv(summary_rows)
    export_json(
        {
            "training": {
                "full_stack_final_reward": full_trace.moving_rewards[-1],
                "rl_only_final_reward": rl_trace.moving_rewards[-1],
                "full_stack_final_score": full_trace.moving_scores[-1],
                "rl_only_final_score": rl_trace.moving_scores[-1],
                "full_stack_final_violation": full_trace.moving_violations[-1],
                "rl_only_final_violation": rl_trace.moving_violations[-1],
            },
            "evaluation": detailed_payload,
        }
    )
    export_table(summary_rows)
    plot_training(full_trace, rl_trace)
    plot_regime_comparison(summary_rows)

    print("Chapter 5 formal experiments completed")
    print(f"- results csv: {CSV_PATH}")
    print(f"- summary json: {JSON_PATH}")
    print(f"- table: {TABLE_PATH}")
    print(f"- training figure: {TRAIN_FIGURE}")
    print(f"- regime figure: {REGIME_FIGURE}")


if __name__ == "__main__":
    main()
