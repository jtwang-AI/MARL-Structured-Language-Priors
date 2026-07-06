from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLORS = {
    "ink": "#263447",
    "input": "#f6f8fb",
    "cognition": "#e8f3ed",
    "decision": "#e7effa",
    "execution": "#fff0e6",
    "eval": "#f3f0f8",
    "accent": "#278c8c",
    "warn": "#b65f16",
    "line": "#6b7280",
}


def add_box(ax, xy, wh, title, lines, facecolor, edgecolor, body_fontsize=7.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.028",
        linewidth=1.45,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.05,
        title,
        ha="center",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    line_y = y + h - 0.116
    for idx, line in enumerate(lines):
        ax.text(
            x + 0.026,
            line_y - idx * 0.05,
            line,
            ha="left",
            va="top",
            fontsize=body_fontsize,
            color=COLORS["ink"],
            zorder=3,
        )


def add_arrow(ax, start, end, text=None, color=None, rad=0.0, text_offset=(0, 0)):
    color = color or COLORS["line"]
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        color=color,
        shrinkA=5,
        shrinkB=5,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mid_x,
            mid_y,
            text,
            ha="center",
            va="center",
            fontsize=6.4,
            color=color,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.94),
            zorder=4,
        )


def add_stage_label(ax, x, text, color):
    ax.text(
        x,
        0.935,
        text,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.08", fc=color, ec="none"),
        zorder=5,
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_stage_label(ax, 0.16, "language / context", "#7b8794")
    add_stage_label(ax, 0.405, "structured priors", "#4f7d5f")
    add_stage_label(ax, 0.655, "safe policy", "#4b6f9e")
    add_stage_label(ax, 0.865, "execution", "#b86c45")

    add_box(
        ax,
        (0.035, 0.555),
        (0.165, 0.300),
        "Mission context",
        ["Text command", "Operation rules", "Sea state, drift", "Team status"],
        COLORS["input"],
        edgecolor="#9aa4b2",
    )
    add_box(
        ax,
        (0.245, 0.515),
        (0.215, 0.375),
        "Structured priors",
        ["Task graph", "Role priors", "Hard constraints", "Reward weights", "Schema validation"],
        COLORS["cognition"],
        edgecolor="#4f7d5f",
    )
    add_box(
        ax,
        (0.505, 0.515),
        (0.220, 0.375),
        "Safe RL decision layer",
        [
            "Prior adapter",
            "State features",
            "Q-learning baseline",
            "Deep actor-critic",
            "Coord. and safety modes",
        ],
        COLORS["decision"],
        edgecolor="#4b6f9e",
        body_fontsize=6.8,
    )
    add_box(
        ax,
        (0.775, 0.515),
        (0.190, 0.375),
        "Execution and safety",
        [
            "Safety projection",
            "Task assignment",
            "Search, verify, protect",
            "UAV-USV simulator",
            "Safe action set",
        ],
        COLORS["execution"],
        edgecolor="#b86c45",
        body_fontsize=6.8,
    )

    add_arrow(ax, (0.200, 0.700), (0.245, 0.700), "parse", COLORS["line"], text_offset=(0, 0.045))
    add_arrow(ax, (0.460, 0.700), (0.505, 0.700), "schema", COLORS["accent"], text_offset=(0, 0.050))
    add_arrow(ax, (0.725, 0.700), (0.775, 0.700), "safe action", COLORS["accent"], text_offset=(0, 0.050))

    add_box(
        ax,
        (0.235, 0.075),
        (0.545, 0.315),
        "Evaluation, robustness, and feedback",
        [
            "Metrics: success, time, formation error,",
            "communication load, violation rate, score",
            "Realism: hydrodynamics, sensing, packet drops",
            "External check: MARL-AUV six-DOF tracking",
        ],
        COLORS["eval"],
        edgecolor="#8064a2",
        body_fontsize=6.8,
    )

    add_arrow(ax, (0.890, 0.515), (0.725, 0.390), "metrics", COLORS["warn"], rad=-0.18, text_offset=(0.105, -0.022))
    add_arrow(ax, (0.550, 0.390), (0.635, 0.515), "reward", COLORS["warn"], rad=0.12, text_offset=(-0.035, 0.0))
    add_arrow(ax, (0.390, 0.390), (0.355, 0.515), "rules", COLORS["warn"], rad=-0.08, text_offset=(-0.045, 0.0))
    add_arrow(ax, (0.695, 0.390), (0.845, 0.515), "stress tests", COLORS["warn"], rad=0.22, text_offset=(-0.025, 0.038))

    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"chapter5_architecture.{fmt}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
