"""Additional robustness experiments for the Chapter 5 small paper."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIM_SRC = ROOT / "simulator"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

import run_extension_experiments as ext  # noqa: E402

from env.chapter4_adapter import ACTIONS, ActionBundle, evaluate_bundle  # noqa: E402
from msar_sim.evaluation import evaluate_method  # noqa: E402
from msar_sim.scenario import generate_scenario  # noqa: E402
from planner.cognitive_adapter import CognitivePrior, load_schema  # noqa: E402
from policy.q_learning import TabularPolicy, evaluate_policy, train_policy  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
SCHEMA_PATH = ROOT / "planner" / "task_graph_schema.json"


SEVERITY_LEVELS = [
    {"label": "nominal", "sea_state": 1.8, "target_speed": 0.9, "communication_quality": 0.88, "regime": "standard", "difficulty": "easy"},
    {"label": "moderate", "sea_state": 3.0, "target_speed": 1.7, "communication_quality": 0.62, "regime": "standard", "difficulty": "medium"},
    {"label": "severe", "sea_state": 4.4, "target_speed": 2.6, "communication_quality": 0.40, "regime": "stress", "difficulty": "hard"},
    {"label": "extreme", "sea_state": 5.4, "target_speed": 3.3, "communication_quality": 0.26, "regime": "stress", "difficulty": "hard"},
]


METHOD_LABELS = {
    "fixed_rule": "Fixed rule",
    "rl_only": "RL only",
    "schema_rule_only": "Schema rule only",
    "full_stack": "Full stack",
}


def train_variant_with_budget(schema: dict, planner_enabled: bool, safety_enabled: bool, seed: int, episodes: int) -> TabularPolicy:
    hard_count = len(schema["hard_constraints"])

    def prior_builder(scenario, enabled: bool) -> CognitivePrior:
        return ext.build_prior(schema, scenario, enabled)

    def evaluator(scenario, action, prior: CognitivePrior, enabled: bool):
        return evaluate_bundle(evaluate_method, scenario, action, prior.reward_weights, hard_count, enabled)

    policy = TabularPolicy(ACTIONS)
    train_policy(policy, ext.scenario_sampler, prior_builder, evaluator, episodes, seed, planner_enabled, safety_enabled)
    return policy


def evaluate_variant(schema: dict, policy: TabularPolicy, scenarios: Sequence, planner_enabled: bool, safety_enabled: bool):
    hard_count = len(schema["hard_constraints"])

    def prior_builder(scenario, enabled: bool) -> CognitivePrior:
        return ext.build_prior(schema, scenario, enabled)

    def evaluator(scenario, action, prior: CognitivePrior, enabled: bool):
        return evaluate_bundle(evaluate_method, scenario, action, prior.reward_weights, hard_count, enabled)

    return evaluate_policy(policy, list(scenarios), prior_builder, evaluator, planner_enabled, safety_enabled)


def summarize(name: str, regime: str, outcomes: Sequence[Dict[str, float]], extra: Dict[str, object] | None = None):
    row = ext.summarize_outcomes(name, regime, outcomes)
    if extra:
        row.update(extra)
    return row


def make_severity_scenarios(level: dict, count: int = 80):
    scenarios = []
    for idx in range(count):
        base = generate_scenario(seed=9000 + idx, scenario_id=idx, regime=level["regime"])
        scenarios.append(
            replace(
                base,
                sea_state=level["sea_state"],
                target_speed=level["target_speed"],
                communication_quality=level["communication_quality"],
                regime=level["regime"],
                difficulty=level["difficulty"],
            )
        )
    return scenarios


def run_severity_sweep(schema: dict) -> List[Dict[str, object]]:
    full_policy = train_variant_with_budget(schema, True, True, 71, 420)
    rl_policy = train_variant_with_budget(schema, False, False, 73, 420)
    hard_count = len(schema["hard_constraints"])
    rows: List[Dict[str, object]] = []
    for level in SEVERITY_LEVELS:
        scenarios = make_severity_scenarios(level)
        fixed = [
            evaluate_bundle(evaluate_method, scenario, ActionBundle("behavior_distributed", "balanced"), ext.neutral_prior(scenario).reward_weights, hard_count, False)
            for scenario in scenarios
        ]
        schema_rule = []
        for scenario in scenarios:
            prior = ext.build_prior(schema, scenario, True)
            schema_rule.append(evaluate_bundle(evaluate_method, scenario, ext.heuristic_action(prior), prior.reward_weights, hard_count, False))
        rl = evaluate_variant(schema, rl_policy, scenarios, False, False)
        full = evaluate_variant(schema, full_policy, scenarios, True, True)
        for name, outcomes in [
            ("fixed_rule", fixed),
            ("rl_only", rl),
            ("schema_rule_only", schema_rule),
            ("full_stack", full),
        ]:
            rows.append(summarize(name, level["regime"], outcomes, {"severity": level["label"]}))
    write_csv(RESULTS_DIR / "severity_sweep_results.csv", rows)
    write_json(RESULTS_DIR / "severity_sweep_summary.json", {"rows": rows})
    export_severity_table(rows)
    plot_severity(rows)
    return rows


def plot_severity(rows: Sequence[Dict[str, object]]) -> None:
    methods = ["fixed_rule", "rl_only", "schema_rule_only", "full_stack"]
    colors = {
        "fixed_rule": "#4c78a8",
        "rl_only": "#355070",
        "schema_rule_only": "#f4a261",
        "full_stack": "#2a9d8f",
    }
    labels = [level["label"] for level in SEVERITY_LEVELS]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    for method in methods:
        method_rows = [next(row for row in rows if row["method"] == method and row["severity"] == label) for label in labels]
        axes[0].plot(x, [row["composite_score"] for row in method_rows], marker="o", linewidth=2.0, color=colors[method], label=METHOD_LABELS[method])
        axes[1].plot(x, [row["violation_rate"] for row in method_rows], marker="o", linewidth=2.0, color=colors[method], label=METHOD_LABELS[method])
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.25, linestyle="--")
        axis.set_xlabel("Risk severity")
    axes[0].set_ylabel("Composite score")
    axes[1].set_ylabel("Violation rate")
    axes[0].set_title("Score under increasing risk")
    axes[1].set_title("Safety under increasing risk")
    handles, labels_plot = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_plot, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "severity_sweep.png", bbox_inches="tight")
    plt.close(fig)


def run_paired_statistics(schema: dict) -> List[Dict[str, object]]:
    full_policy = train_variant_with_budget(schema, True, True, 81, 420)
    rl_policy = train_variant_with_budget(schema, False, False, 83, 420)
    scenarios = [generate_scenario(seed=11000 + idx, scenario_id=idx, regime="stress") for idx in range(120)]
    hard_count = len(schema["hard_constraints"])
    fixed = [
        evaluate_bundle(evaluate_method, scenario, ActionBundle("behavior_distributed", "balanced"), ext.neutral_prior(scenario).reward_weights, hard_count, False)
        for scenario in scenarios
    ]
    rl = evaluate_variant(schema, rl_policy, scenarios, False, False)
    full = evaluate_variant(schema, full_policy, scenarios, True, True)
    rows = []
    comparisons = {
        "full_vs_fixed": (full, fixed),
        "full_vs_rl": (full, rl),
    }
    metric_specs = [
        ("composite_score", "higher"),
        ("rescue_success", "higher"),
        ("formation_error", "lower"),
        ("violation_rate", "lower"),
        ("communication_load", "lower"),
    ]
    for comparison, (left, right) in comparisons.items():
        for metric, direction in metric_specs:
            deltas = [l[metric] - r[metric] for l, r in zip(left, right)]
            if direction == "lower":
                deltas = [-delta for delta in deltas]
            ci_low, ci_high = bootstrap_ci(deltas)
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "direction": direction,
                    "mean_improvement": mean(deltas),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "sign_test_p": sign_test_pvalue(deltas),
                    "win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
                }
            )
    write_csv(RESULTS_DIR / "paired_statistics.csv", rows)
    write_json(RESULTS_DIR / "paired_statistics_summary.json", {"rows": rows, "n": len(scenarios)})
    export_statistics_table(rows)
    return rows


def bootstrap_ci(values: Sequence[float], iters: int = 2000, seed: int = 19):
    rng = Random(seed)
    samples = []
    n = len(values)
    for _ in range(iters):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    return samples[int(0.025 * (iters - 1))], samples[int(0.975 * (iters - 1))]


def sign_test_pvalue(values: Sequence[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    tail_count = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return min(1.0, 2.0 * tail_count / (2**n))


def run_training_budget(schema: dict) -> List[Dict[str, object]]:
    budgets = [120, 240, 420, 720]
    scenarios = [generate_scenario(seed=13000 + idx, scenario_id=idx, regime="stress") for idx in range(80)]
    rows = []
    for budget in budgets:
        full_policy = train_variant_with_budget(schema, True, True, 90 + budget, budget)
        rl_policy = train_variant_with_budget(schema, False, False, 190 + budget, budget)
        full = evaluate_variant(schema, full_policy, scenarios, True, True)
        rl = evaluate_variant(schema, rl_policy, scenarios, False, False)
        rows.append(summarize("full_stack", "stress", full, {"episodes": budget}))
        rows.append(summarize("rl_only", "stress", rl, {"episodes": budget}))
    write_csv(RESULTS_DIR / "training_budget_results.csv", rows)
    write_json(RESULTS_DIR / "training_budget_summary.json", {"rows": rows})
    export_training_budget_table(rows)
    plot_training_budget(rows)
    return rows


def plot_training_budget(rows: Sequence[Dict[str, object]]) -> None:
    budgets = sorted({int(row["episodes"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    for method, color in [("rl_only", "#355070"), ("full_stack", "#2a9d8f")]:
        method_rows = [next(row for row in rows if row["method"] == method and int(row["episodes"]) == budget) for budget in budgets]
        axes[0].plot(budgets, [row["composite_score"] for row in method_rows], marker="o", linewidth=2.0, color=color, label=METHOD_LABELS[method])
        axes[1].plot(budgets, [row["violation_rate"] for row in method_rows], marker="o", linewidth=2.0, color=color, label=METHOD_LABELS[method])
    for axis in axes:
        axis.grid(alpha=0.25, linestyle="--")
        axis.set_xlabel("Training episodes")
    axes[0].set_ylabel("Stress composite score")
    axes[1].set_ylabel("Stress violation rate")
    axes[0].set_title("Training-budget sensitivity")
    axes[1].set_title("Safety vs. training budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "training_budget_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


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


def export_severity_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Risk-severity sweep. Higher score and lower violation are better.}",
        "  \\label{tab:severity_sweep}",
        "  \\begin{tabular}{llrrr}",
        "    \\toprule",
        "    Severity & Method & Rescue & Violation & Score \\\\",
        "    \\midrule",
    ]
    for severity in ["nominal", "moderate", "severe", "extreme"]:
        for method in ["fixed_rule", "rl_only", "schema_rule_only", "full_stack"]:
            row = next(item for item in rows if item["severity"] == severity and item["method"] == method)
            lines.append(
                f"    {severity.capitalize()} & {METHOD_LABELS[method]} & {row['rescue_success']:.3f} & {row['violation_rate']:.3f} & {row['composite_score']:.2f} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "severity_sweep_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_statistics_table(rows: Sequence[Dict[str, object]]) -> None:
    selected = [
        row
        for row in rows
        if row["comparison"] in {"full_vs_fixed", "full_vs_rl"}
        and row["metric"] in {"composite_score", "formation_error", "violation_rate"}
    ]
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Paired stress-regime improvements with bootstrap confidence intervals.}",
        "  \\label{tab:paired_statistics}",
        "  \\begin{tabular}{llrrrr}",
        "    \\toprule",
        "    Comparison & Metric & Mean impr. & 95\\% CI low & 95\\% CI high & Win rate \\\\",
        "    \\midrule",
    ]
    for row in selected:
        comparison = "Full vs. fixed" if row["comparison"] == "full_vs_fixed" else "Full vs. RL"
        metric = row["metric"].replace("_", " ")
        lines.append(
            f"    {comparison} & {metric} & {row['mean_improvement']:.3f} & {row['ci_low']:.3f} & {row['ci_high']:.3f} & {row['win_rate']:.3f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "paired_statistics_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_training_budget_table(rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Training-budget sensitivity in stress regimes.}",
        "  \\label{tab:training_budget}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Method / Episodes & Rescue & Violation & Score \\\\",
        "    \\midrule",
    ]
    for budget in [120, 240, 420, 720]:
        for method in ["rl_only", "full_stack"]:
            row = next(item for item in rows if int(item["episodes"]) == budget and item["method"] == method)
            lines.append(
                f"    {METHOD_LABELS[method]} / {budget} & {row['rescue_success']:.3f} & {row['violation_rate']:.3f} & {row['composite_score']:.2f} \\\\"
            )
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines.extend(["  \\end{tabular}", "\\end{table}"])
    (RESULTS_DIR / "training_budget_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    severity_rows = run_severity_sweep(schema)
    stats_rows = run_paired_statistics(schema)
    budget_rows = run_training_budget(schema)
    write_json(
        RESULTS_DIR / "additional_experiments_summary.json",
        {
            "severity_sweep": severity_rows,
            "paired_statistics": stats_rows,
            "training_budget": budget_rows,
        },
    )
    print("Additional experiments completed")
    print(f"- severity sweep: {RESULTS_DIR / 'severity_sweep_results.csv'}")
    print(f"- paired statistics: {RESULTS_DIR / 'paired_statistics.csv'}")
    print(f"- training budget: {RESULTS_DIR / 'training_budget_results.csv'}")


if __name__ == "__main__":
    main()
