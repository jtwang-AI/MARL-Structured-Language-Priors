import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt

from .evaluation import ABLATIONS, METHODS, evaluate_method
from .scenario import generate_scenario


METRIC_FIELDS = [
    "detection_time",
    "verification_time",
    "encirclement_time",
    "mission_time",
    "total_path_length",
    "utility_score",
    "formation_error",
    "rescue_success",
    "communication_load",
]

METHOD_LABELS = {
    "behavior_distributed": "Proposed",
    "homogeneous_distributed": "Homogeneous",
    "distance_only": "Distance-only",
    "centralized_optimal": "Centralized",
}

METHOD_COLORS = {
    "behavior_distributed": "#0f6cbd",
    "homogeneous_distributed": "#00a36c",
    "distance_only": "#f39c12",
    "centralized_optimal": "#c0392b",
}


def run_batch(
    num_scenarios: int,
    output_dir: Path,
    seed: int = 42,
    regime: str = "standard",
    include_ablations: bool = False,
) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_idx in range(num_scenarios):
        scenario = generate_scenario(
            seed=seed + scenario_idx,
            scenario_id=scenario_idx,
            regime=regime,
        )
        for method in METHODS:
            rows.append(asdict(evaluate_method(scenario, method)))
        if include_ablations:
            for ablation in [name for name in ABLATIONS if name != "full"]:
                rows.append(asdict(evaluate_method(scenario, "behavior_distributed", ablation=ablation)))

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    difficulty_summary = summarize_by_difficulty(rows)
    difficulty_path = output_dir / "difficulty_summary.json"
    with difficulty_path.open("w", encoding="utf-8") as handle:
        json.dump(difficulty_summary, handle, indent=2)

    significance = compute_significance(rows)
    significance_path = output_dir / "significance_summary.json"
    with significance_path.open("w", encoding="utf-8") as handle:
        json.dump(significance, handle, indent=2)

    distribution_summary = summarize_distributions(rows)
    with (output_dir / "distribution_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(distribution_summary, handle, indent=2)

    win_summary = summarize_wins(rows)
    with (output_dir / "wins_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(win_summary, handle, indent=2)

    sensitivity_summary = summarize_sensitivity(rows)
    with (output_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(sensitivity_summary, handle, indent=2)

    regime_summary = summarize_by_regime(rows)
    with (output_dir / "regime_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(regime_summary, handle, indent=2)
    ablation_summary = summarize_ablation(rows)
    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(ablation_summary, handle, indent=2)
    export_latex_tables(
        summary,
        difficulty_summary,
        regime_summary,
        ablation_summary,
        distribution_summary,
        win_summary,
        output_dir,
    )

    plot_summary(summary, output_dir)
    plot_distributions(rows, output_dir)
    plot_cdf(rows, output_dir, metric_key="mission_time", title="Mission-Time CDF", filename="mission_time_cdf.png")
    plot_sensitivity(rows, output_dir, x_key="sea_state", y_key="mission_time", xlabel="Sea-State Level", filename="mission_time_vs_sea_state.png")
    plot_sensitivity(rows, output_dir, x_key="communication_quality", y_key="mission_time", xlabel="Communication Quality", filename="mission_time_vs_communication.png")
    plot_wins(win_summary, output_dir)
    plot_ablation(ablation_summary, output_dir)
    return summary


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: mean(float(row[metric]) for row in method_rows)
            for metric in METRIC_FIELDS
        }
    return summary


def summarize_by_difficulty(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    methods = sorted({row["method"] for row in rows})
    difficulties = sorted({row["difficulty"] for row in rows})
    for difficulty in difficulties:
        summary[difficulty] = {}
        for method in methods:
            method_rows = [
                row for row in rows if row["method"] == method and row["difficulty"] == difficulty
            ]
            if not method_rows:
                continue
            summary[difficulty][method] = {
                metric: mean(float(row[metric]) for row in method_rows)
                for metric in METRIC_FIELDS
            }
    return summary


def summarize_by_regime(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    methods = sorted({row["method"] for row in rows})
    regimes = sorted({row["regime"] for row in rows})
    for regime in regimes:
        summary[regime] = {}
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method and row["regime"] == regime]
            if not method_rows:
                continue
            summary[regime][method] = {
                metric: mean(float(row[metric]) for row in method_rows)
                for metric in METRIC_FIELDS
            }
    return summary


def summarize_ablation(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    targets = [row for row in rows if row["method"].startswith("behavior_distributed__")]
    methods = sorted({row["method"] for row in targets})
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in targets if row["method"] == method]
        summary[method] = {
            metric: mean(float(row[metric]) for row in method_rows)
            for metric in METRIC_FIELDS
        }
    return summary


def summarize_distributions(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = ["mission_time", "rescue_success", "formation_error"]
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in sorted({row["method"] for row in rows if "__" not in row["method"]}):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for metric in metrics:
            values = sorted(float(row[metric]) for row in method_rows)
            count = len(values)
            if count == 0:
                continue
            q1 = values[max(0, int(0.25 * (count - 1)))]
            q3 = values[max(0, int(0.75 * (count - 1)))]
            summary[method][metric] = {
                "mean": mean(values),
                "median": median(values),
                "min": values[0],
                "max": values[-1],
                "q1": q1,
                "q3": q3,
            }
    return summary


def summarize_wins(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    relevant_rows = [row for row in rows if "__" not in row["method"]]
    scenario_ids = sorted({int(row["scenario_id"]) for row in relevant_rows})
    methods = sorted({row["method"] for row in relevant_rows})
    win_counts = {method: 0 for method in methods}
    rescue_wins = {method: 0 for method in methods}

    for scenario_id in scenario_ids:
        scenario_rows = [row for row in relevant_rows if int(row["scenario_id"]) == scenario_id]
        best_time = min(scenario_rows, key=lambda row: float(row["mission_time"]))["method"]
        best_rescue = max(scenario_rows, key=lambda row: float(row["rescue_success"]))["method"]
        win_counts[best_time] += 1
        rescue_wins[best_rescue] += 1

    total = max(len(scenario_ids), 1)
    return {
        method: {
            "mission_time_wins": float(win_counts[method]),
            "mission_time_win_rate": win_counts[method] / total,
            "rescue_success_wins": float(rescue_wins[method]),
            "rescue_success_win_rate": rescue_wins[method] / total,
        }
        for method in methods
    }


def summarize_sensitivity(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    relevant_rows = [row for row in rows if "__" not in row["method"]]
    specs = {
        "sea_state": [(0.0, 2.0, "low"), (2.0, 3.5, "moderate"), (3.5, 10.0, "high")],
        "communication_quality": [(0.0, 0.45, "poor"), (0.45, 0.7, "degraded"), (0.7, 1.01, "strong")],
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, bins in specs.items():
        summary[key] = {}
        for lower, upper, label in bins:
            bucket_rows = [row for row in relevant_rows if lower <= float(row[key]) < upper]
            if not bucket_rows:
                continue
            summary[key][label] = {}
            for method in sorted({row["method"] for row in bucket_rows}):
                method_rows = [row for row in bucket_rows if row["method"] == method]
                summary[key][label][method] = {
                    "mission_time": mean(float(row["mission_time"]) for row in method_rows),
                    "rescue_success": mean(float(row["rescue_success"]) for row in method_rows),
                    "formation_error": mean(float(row["formation_error"]) for row in method_rows),
                }
    return summary


def export_latex_tables(
    summary: Dict[str, Dict[str, float]],
    difficulty_summary: Dict[str, Dict[str, Dict[str, float]]],
    regime_summary: Dict[str, Dict[str, Dict[str, float]]],
    ablation_summary: Dict[str, Dict[str, float]],
    distribution_summary: Dict[str, Dict[str, Dict[str, float]]],
    win_summary: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    main_table = output_dir / "results_table.tex"
    with main_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Mission Time & Rescue Success & Path Len. & Comm.\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in summary.items():
            handle.write(
                f"{method.replace('_', ' ')} & {metrics['mission_time']:.2f} & "
                f"{metrics['rescue_success']:.3f} & {metrics['total_path_length']:.2f} & "
                f"{metrics['communication_load']:.2f}\\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")

    distribution_table = output_dir / "distribution_table.tex"
    with distribution_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lccc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Median $T_{\\mathrm{mis}}$ & IQR $T_{\\mathrm{mis}}$ & Median $S_{\\mathrm{res}}$\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in distribution_summary.items():
            mission = metrics["mission_time"]
            rescue = metrics["rescue_success"]
            iqr = mission["q3"] - mission["q1"]
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {mission['median']:.2f} & "
                f"{iqr:.2f} & {rescue['median']:.3f}\\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")

    wins_table = output_dir / "wins_table.tex"
    with wins_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Mission-time wins & Rescue-success wins\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in win_summary.items():
            handle.write(
                f"{METHOD_LABELS.get(method, method)} & {metrics['mission_time_wins']:.0f} & "
                f"{metrics['rescue_success_wins']:.0f}\\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")

    difficulty_table = output_dir / "difficulty_table.tex"
    with difficulty_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcc}\n")
        handle.write("\\toprule\n")
        handle.write("Difficulty & Method & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for difficulty, method_data in difficulty_summary.items():
            for method, metrics in method_data.items():
                handle.write(
                    f"{difficulty} & {method.replace('_', ' ')} & "
                    f"{metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n"
                )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")

    regime_table = output_dir / "regime_table.tex"
    with regime_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llcc}\n")
        handle.write("\\toprule\n")
        handle.write("Regime & Method & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for regime, method_data in regime_summary.items():
            for method, metrics in method_data.items():
                handle.write(
                    f"{regime} & {method.replace('_', ' ')} & "
                    f"{metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n"
                )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")

    ablation_table = output_dir / "ablation_table.tex"
    with ablation_table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n")
        handle.write("\\toprule\n")
        handle.write("Ablation & Mission Time & Rescue Success\\\\\n")
        handle.write("\\midrule\n")
        for method, metrics in ablation_summary.items():
            label = method.replace("behavior_distributed__", "").replace("_", " ")
            handle.write(f"{label} & {metrics['mission_time']:.2f} & {metrics['rescue_success']:.3f}\\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular*}\n")


def compute_significance(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    methods = sorted({row["method"] for row in rows})
    metrics = ["mission_time", "rescue_success"]
    result: Dict[str, Dict[str, float]] = {}
    baseline_rows = [row for row in rows if row["method"] == "behavior_distributed"]
    for method in methods:
        if method == "behavior_distributed":
            continue
        compare_rows = [row for row in rows if row["method"] == method]
        result[method] = {}
        for metric in metrics:
            deltas = [
                float(base[metric]) - float(other[metric])
                for base, other in zip(baseline_rows, compare_rows)
            ]
            avg = mean(deltas)
            std = pstdev(deltas) if len(deltas) > 1 else 0.0
            result[method][f"{metric}_delta_mean"] = avg
            result[method][f"{metric}_delta_std"] = std
    return result


def plot_summary(summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    metrics = [
        ("mission_time", "Mission Time"),
        ("rescue_success", "Rescue Success"),
        ("total_path_length", "Path Length"),
        ("communication_load", "Communication Load"),
    ]
    methods = list(summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (metric_key, title) in zip(axes.flatten(), metrics):
        values = [summary[method][metric_key] for method in methods]
        axis.bar(
            [METHOD_LABELS.get(method, method) for method in methods],
            values,
            color=[METHOD_COLORS.get(method, "#777777") for method in methods],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=200)
    plt.close(fig)


def plot_distributions(rows: List[Dict[str, float]], output_dir: Path) -> None:
    methods = [method for method in sorted({row["method"] for row in rows}) if "__" not in method]
    metrics = [("mission_time", "Mission Time"), ("rescue_success", "Rescue Success")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (metric_key, title) in zip(axes, metrics):
        data = [
            [float(row[metric_key]) for row in rows if row["method"] == method]
            for method in methods
        ]
        bp = axis.boxplot(
            data,
            patch_artist=True,
            tick_labels=[METHOD_LABELS.get(method, method) for method in methods],
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "#777777"))
            patch.set_alpha(0.65)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution_boxplots.png", dpi=220)
    plt.close(fig)


def plot_cdf(
    rows: List[Dict[str, float]],
    output_dir: Path,
    metric_key: str,
    title: str,
    filename: str,
) -> None:
    methods = [method for method in sorted({row["method"] for row in rows}) if "__" not in method]
    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    for method in methods:
        values = sorted(float(row[metric_key]) for row in rows if row["method"] == method)
        total = len(values)
        ys = [(idx + 1) / total for idx in range(total)]
        axis.plot(values, ys, linewidth=2.0, color=METHOD_COLORS.get(method, "#777777"), label=METHOD_LABELS.get(method, method))
    axis.set_title(title)
    axis.set_xlabel(metric_key.replace("_", " ").title())
    axis.set_ylabel("Empirical CDF")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def plot_sensitivity(
    rows: List[Dict[str, float]],
    output_dir: Path,
    x_key: str,
    y_key: str,
    xlabel: str,
    filename: str,
) -> None:
    methods = [method for method in sorted({row["method"] for row in rows}) if "__" not in method]
    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        xs = sorted(float(row[x_key]) for row in method_rows)
        x_min, x_max = xs[0], xs[-1]
        width = max((x_max - x_min) / 3.0, 1e-6)
        centers = []
        means = []
        start = x_min
        for idx in range(3):
            lower = start + idx * width
            upper = x_max + 1e-6 if idx == 2 else lower + width
            bucket = [float(row[y_key]) for row in method_rows if lower <= float(row[x_key]) <= upper]
            if not bucket:
                continue
            centers.append((lower + upper) / 2.0)
            means.append(mean(bucket))
        if centers:
            axis.plot(centers, means, marker="o", linewidth=2.0, color=METHOD_COLORS.get(method, "#777777"), label=METHOD_LABELS.get(method, method))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(y_key.replace("_", " ").title())
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def plot_wins(win_summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    methods = list(win_summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    mission_values = [win_summary[method]["mission_time_wins"] for method in methods]
    rescue_values = [win_summary[method]["rescue_success_wins"] for method in methods]
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    colors = [METHOD_COLORS.get(method, "#777777") for method in methods]
    axes[0].bar(labels, mission_values, color=colors)
    axes[0].set_title("Mission-Time Wins")
    axes[1].bar(labels, rescue_values, color=colors)
    axes[1].set_title("Rescue-Success Wins")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "win_counts.png", dpi=220)
    plt.close(fig)


def plot_ablation(ablation_summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not ablation_summary:
        return
    methods = list(ablation_summary.keys())
    labels = [method.replace("behavior_distributed__", "").replace("_", " ") for method in methods]
    mission_values = [ablation_summary[method]["mission_time"] for method in methods]
    rescue_values = [ablation_summary[method]["rescue_success"] for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, mission_values, color="#0f6cbd")
    axes[0].set_title("Ablation: Mission Time")
    axes[1].bar(labels, rescue_values, color="#00a36c")
    axes[1].set_title("Ablation: Rescue Success")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_bar.png", dpi=220)
    plt.close(fig)
