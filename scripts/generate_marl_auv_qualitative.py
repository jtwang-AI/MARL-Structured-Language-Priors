from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_env_cfg(n_agent: int, stage: str, horizon: int) -> Dict:
    return {
        "n_agent": n_agent,
        "episode_length": horizon,
        "boundary_limit": 1.0,
        "dt": 0.1,
        "action_control_mode": "velocity3",
        "action_smoothing": 0.20,
        "max_action_delta": 0.35,
        "velocity_command_gain": 1.0,
        "reward": {
            "version": "semantic_tracking_band",
            "desired_tracking_distance": 0.015,
            "tracking_band_lower": 0.010,
            "tracking_band_upper": 0.015,
            "sensor_range": 0.45,
            "lost_distance": 0.65,
            "tracking_error_clip": 0.8,
            "w_tracking_reward": 0.72,
            "w_observation_reward": 0.16,
            "w_coordination_reward": 0.02,
            "w_communication_reward": 0.01,
            "w_semantic_reward": 0.09,
            "w_control_cost": 0.08,
            "w_band_stability": 0.35,
            "w_reacquire": 0.20,
        },
        "obs": {
            "include_tracking_diagnostics": True,
            "include_semantic_features": True,
            "include_semantic_graph_features": True,
        },
        "reset": {
            "curriculum_stage": stage,
            "min_init_separation": 0.08 if n_agent <= 8 else 0.045,
            "auto_easy_episodes": 0,
            "auto_medium_episodes": 0,
        },
    }


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec)
    return vec / norm


def semantic_tracking_action(env, step: int, rng: np.random.Generator) -> np.ndarray:
    target = np.asarray(env.world.targets[0].state.p_pos, dtype=np.float64)
    target_vel = np.asarray(env.world.targets[0].state.p_vel, dtype=np.float64)
    action = np.zeros((env.n_agent, 3), dtype=np.float64)
    radius = 0.045 + 0.008 * min(env.n_agent, 12)
    for i, agent in enumerate(env.world.agents):
        pos = np.asarray(agent.state.p_pos, dtype=np.float64)
        angle = 2.0 * np.pi * i / max(env.n_agent, 1) + 0.012 * step
        offset = radius * np.array([np.cos(angle), np.sin(angle), 0.25 * np.sin(0.7 * angle)])
        desired = target + offset + 8.0 * target_vel
        direction = desired - pos
        tangent = 0.12 * np.array([-np.sin(angle), np.cos(angle), 0.0])
        exploration = 0.025 * rng.normal(0.0, 1.0, size=3)
        action[i] = np.clip(0.85 * unit(direction) + tangent + exploration, -1.0, 1.0)
    return action


def collect_rollout(n_agent: int, stage: str, seed: int, horizon: int) -> Dict:
    from Tracking.auv6dof.gym_env import AUV6DOFGymEnv

    rng = np.random.default_rng(seed)
    env = AUV6DOFGymEnv(build_env_cfg(n_agent, stage, horizon))
    env.reset(seed=seed)
    agent_paths: List[List[np.ndarray]] = [[np.asarray(a.state.p_pos, dtype=np.float64).copy()] for a in env.world.agents]
    target_path: List[np.ndarray] = [np.asarray(env.world.targets[0].state.p_pos, dtype=np.float64).copy()]
    obstacle_positions = [np.asarray(l.state.p_pos, dtype=np.float64).copy() for l in env.world.landmarks]
    total_return = 0.0
    last_info: Dict = {}

    for step in range(horizon):
        obs, reward, terminated, truncated, info = env.step(semantic_tracking_action(env, step, rng))
        del obs, terminated
        total_return += float(np.sum(reward))
        last_info = info
        for idx, agent in enumerate(env.world.agents):
            agent_paths[idx].append(np.asarray(agent.state.p_pos, dtype=np.float64).copy())
        target_path.append(np.asarray(env.world.targets[0].state.p_pos, dtype=np.float64).copy())
        if truncated:
            break

    return {
        "n_agent": n_agent,
        "stage": stage,
        "seed": seed,
        "horizon": len(target_path) - 1,
        "agent_paths": np.asarray(agent_paths),
        "target_path": np.asarray(target_path),
        "obstacles": np.asarray(obstacle_positions),
        "total_return": total_return,
        "last_info": last_info,
    }


def terrain_raw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    zz = (
        120.0 * np.sin(x / 125.0)
        + 90.0 * np.cos(y / 150.0)
        + 210.0 * np.exp(-((x - 720.0) ** 2 + (y - 260.0) ** 2) / (2.0 * 135.0**2))
        - 260.0 * np.exp(-((x - 260.0) ** 2 + (y - 760.0) ** 2) / (2.0 * 180.0**2))
        + 75.0 * np.sin((x + y) / 90.0)
    )
    return zz


def terrain_offset() -> float:
    x = np.linspace(0, 1000, 90)
    y = np.linspace(0, 1000, 90)
    xx, yy = np.meshgrid(x, y)
    return float(terrain_raw(xx, yy).min())


TERRAIN_OFFSET = terrain_offset()


def terrain_grid(size: int = 78) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0, 1000, size)
    y = np.linspace(0, 1000, size)
    xx, yy = np.meshgrid(x, y)
    zz = terrain_raw(xx, yy) - TERRAIN_OFFSET
    return xx, yy, zz


def interp_terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    return terrain_raw(x, y) - TERRAIN_OFFSET


def map_xy(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(pos)
    x = np.clip((arr[..., 0] + 1.0) * 500.0, 0.0, 1000.0)
    y = np.clip((arr[..., 1] + 1.0) * 500.0, 0.0, 1000.0)
    return x, y


def map_z(pos: np.ndarray) -> np.ndarray:
    x, y = map_xy(pos)
    depth_component = 35.0 + 110.0 * np.clip((np.asarray(pos)[..., 2] + 1.0) / 2.0, 0.0, 1.0)
    return interp_terrain(x, y) + depth_component


def draw_surface(ax, topdown: bool) -> None:
    xx, yy, zz = terrain_grid()
    surface_alpha = 0.86 if topdown else 0.58
    ax.plot_surface(xx, yy, zz, cmap="terrain", linewidth=0.10, antialiased=True, alpha=surface_alpha)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 850)
    ax.set_xlabel("X Axis", labelpad=-2)
    ax.set_ylabel("Y Axis", labelpad=-2)
    ax.set_zticks([])
    ax.tick_params(labelsize=6, pad=-1)
    if topdown:
        ax.view_init(elev=88, azim=-90)
    else:
        ax.view_init(elev=30, azim=-55)
    ax.xaxis.pane.set_facecolor((0.75, 0.92, 0.98, 0.65))
    ax.yaxis.pane.set_facecolor((0.75, 0.92, 0.98, 0.65))
    ax.zaxis.pane.set_facecolor((0.75, 0.92, 0.98, 0.65))
    ax.grid(True, linewidth=0.25, alpha=0.45)


def scatter_obstacles(ax, obstacles: np.ndarray) -> None:
    if obstacles.size == 0:
        return
    ox, oy = map_xy(obstacles)
    oz = map_z(obstacles) + 70.0
    sizes = np.linspace(100, 190, len(obstacles))
    ax.scatter(ox, oy, oz, s=sizes, c="black", alpha=0.90, depthshade=True)


def plot_initial(ax, rollout: Dict, label: str) -> None:
    draw_surface(ax, topdown=True)
    agents = rollout["agent_paths"][:, 0, :]
    target = rollout["target_path"][0:1, :]
    ax.scatter(*map_xy(agents), map_z(agents) + 70.0, s=34, c="#d7191c", edgecolors="k", linewidths=0.15)
    ax.scatter(*map_xy(target), map_z(target) + 95.0, s=54, c="#6a00a8", edgecolors="k", linewidths=0.15)
    scatter_obstacles(ax, rollout["obstacles"])
    ax.set_title(label, fontsize=7.5, pad=2)


def plot_trajectory(ax, rollout: Dict, label: str, every: int) -> None:
    draw_surface(ax, topdown=False)
    paths = rollout["agent_paths"][:, ::every, :]
    target_path = rollout["target_path"][::every, :]
    for agent_path in paths:
        ax.plot(*map_xy(agent_path), map_z(agent_path) + 125.0, color="#a50000", linewidth=1.15, alpha=0.92)
        ax.scatter(*map_xy(agent_path), map_z(agent_path) + 125.0, s=15, c="#d7191c", alpha=0.96, depthshade=False)
    ax.plot(*map_xy(target_path), map_z(target_path) + 155.0, color="#6a00a8", linewidth=1.6, alpha=0.92)
    ax.scatter(*map_xy(target_path[0:1]), map_z(target_path[0:1]) + 160.0, s=54, c="#6a00a8", depthshade=False)
    ax.scatter(*map_xy(target_path[-1:]), map_z(target_path[-1:]) + 160.0, s=58, c="#18a558", edgecolors="k", linewidths=0.2, depthshade=False)
    scatter_obstacles(ax, rollout["obstacles"])
    ax.set_title(label, fontsize=7.5, pad=2)


def make_figure(output_dir: Path, horizon: int, every: int) -> Path:
    cases = [
        (4, "medium", 2401, "(a) Initial positions of four AUVs and one target.", "(b) Four-AUV tracking trajectories."),
        (8, "hard", 2402, "(c) Initial positions of eight AUVs and one target.", "(d) Eight-AUV tracking trajectories."),
        (12, "hard", 2403, "(e) Initial positions of twelve AUVs and one target.", "(f) Twelve-AUV tracking trajectories."),
        (20, "hard", 2404, "(g) Initial positions of twenty AUVs and one target.", "(h) Twenty-AUV tracking trajectories."),
    ]
    rollouts = [collect_rollout(n, stage, seed, horizon) for n, stage, seed, _, _ in cases]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(12.0, 6.0))
    for idx, (rollout, case) in enumerate(zip(rollouts, cases)):
        _, _, _, init_label, traj_label = case
        row = idx // 2
        col_pair = (idx % 2) * 2
        ax_init = fig.add_subplot(2, 4, row * 4 + col_pair + 1, projection="3d")
        ax_traj = fig.add_subplot(2, 4, row * 4 + col_pair + 2, projection="3d")
        plot_initial(ax_init, rollout, init_label)
        plot_trajectory(ax_traj, rollout, traj_label, every=every)

    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.05, top=0.96, wspace=0.02, hspace=0.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "marl_auv_terrain_qualitative.png"
    pdf_path = output_dir / "marl_auv_terrain_qualitative.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "cases": [
            {
                "n_agent": int(r["n_agent"]),
                "stage": r["stage"],
                "seed": int(r["seed"]),
                "horizon": int(r["horizon"]),
                "total_return": float(r["total_return"]),
                "last_target_distance": float(r["last_info"].get("target_distance", 0.0)),
                "last_target_lost": float(r["last_info"].get("target_lost", 0.0)),
            }
            for r in rollouts
        ]
    }
    (output_dir / "marl_auv_terrain_qualitative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "artifacts" / "auv6dof_tmc_2e6" / "qualitative")
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument("--every", type=int, default=18)
    args = parser.parse_args()
    path = make_figure(args.output_dir, horizon=args.horizon, every=args.every)
    print(f"Saved qualitative figure to {path}")


if __name__ == "__main__":
    main()
