"""Sensitivity analysis for the reported composite score in Equation (4)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.chapter4_adapter import COMPOSITE_SCORE_WEIGHTS, compute_composite_score  # noqa: E402


RESULTS_DIR = ROOT / "results"
FORMAL_RESULTS = RESULTS_DIR / "formal_results.csv"

METHOD_LABELS = {
    "fixed_rule": "Fixed rule",
    "rl_only": "RL only",
    "llm_rule_only": "Schema rule",
    "full_stack": "Full stack",
}

WEIGHT_PROFILES = {
    "nominal": {
        "label": "Nominal",
        "weights": COMPOSITE_SCORE_WEIGHTS,
    },
    "rescue_heavy": {
        "label": "Rescue-heavy",
        "weights": {
            **COMPOSITE_SCORE_WEIGHTS,
            "rescue_success": 160.0,
        },
    },
    "safety_heavy": {
        "label": "Safety-heavy",
        "weights": {
            **COMPOSITE_SCORE_WEIGHTS,
            "violation_rate": 30.0,
            "formation_error": 0.50,
        },
    },
    "communication_heavy": {
        "label": "Comm-heavy",
        "weights": {
            **COMPOSITE_SCORE_WEIGHTS,
            "communication_load": 0.35,
        },
    },
    "speed_first": {
        "label": "Speed-first",
        "weights": {
            **COMPOSITE_SCORE_WEIGHTS,
            "mission_time": 0.65,
        },
    },
}


def load_formal_rows() -> List[Dict[str, float | str]]:
    with FORMAL_RESULTS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float | str]] = []
        for row in reader:
            parsed: Dict[str, float | str] = {"method": row["method"], "regime": row["regime"]}
            for key, value in row.items():
                if key not in {"method", "regime"}:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def weight_vector(weights: Dict[str, float]) -> str:
    return "({rescue_success:.0f},{violation_rate:.0f},{mission_time:.2f},{formation_error:.2f},{communication_load:.2f})".format(
        **weights
    )


def score_row(row: Dict[str, float | str], weights: Dict[str, float]) -> float:
    metrics = {
        "rescue_success": float(row["rescue_success"]),
        "mission_time": float(row["mission_time"]),
        "formation_error": float(row["formation_error"]),
        "communication_load": float(row["communication_load"]),
    }
    return compute_composite_score(metrics, float(row["violation_rate"]), weights)


def run_sensitivity() -> List[Dict[str, object]]:
    stress_rows = [row for row in load_formal_rows() if row["regime"] == "stress"]
    rows: List[Dict[str, object]] = []
    for profile_key, profile in WEIGHT_PROFILES.items():
        weights = dict(profile["weights"])
        method_scores = {
            str(row["method"]): score_row(row, weights)
            for row in stress_rows
        }
        best_method = max(method_scores, key=method_scores.get)
        rows.append(
            {
                "profile": profile_key,
                "label": profile["label"],
                "weights": weight_vector(weights),
                "fixed_rule_score": method_scores["fixed_rule"],
                "rl_only_score": method_scores["rl_only"],
                "schema_rule_score": method_scores["llm_rule_only"],
                "full_stack_score": method_scores["full_stack"],
                "full_minus_fixed": method_scores["full_stack"] - method_scores["fixed_rule"],
                "full_minus_rl": method_scores["full_stack"] - method_scores["rl_only"],
                "best_method": best_method,
                "best_label": METHOD_LABELS[best_method],
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_table(rows: List[Dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Sensitivity of stress-regime conclusions to Equation~(4) weights. Weight vector order is $(w_p,w_v,w_T,w_e,w_c)$.}",
        "  \\label{tab:reward_sensitivity}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llrrrrr}",
        "    \\toprule",
        "    Profile & Weights & Fixed & RL only & Schema rule & Full stack & Best \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    {label} & {weights} & {fixed_rule_score:.2f} & {rl_only_score:.2f} & {schema_rule_score:.2f} & {full_stack_score:.2f} & {best_label} \\\\".format(
                **row
            )
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}}",
            "\\end{table}",
        ]
    )
    (RESULTS_DIR / "reward_sensitivity_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_sensitivity()
    write_csv(RESULTS_DIR / "reward_sensitivity_results.csv", rows)
    (RESULTS_DIR / "reward_sensitivity_summary.json").write_text(
        json.dumps({"source": str(FORMAL_RESULTS.relative_to(ROOT)), "profiles": rows}, indent=2),
        encoding="utf-8",
    )
    export_table(rows)
    print("Reward sensitivity analysis completed")
    print(f"- results: {RESULTS_DIR / 'reward_sensitivity_results.csv'}")
    print(f"- table: {RESULTS_DIR / 'reward_sensitivity_table.tex'}")


if __name__ == "__main__":
    main()
