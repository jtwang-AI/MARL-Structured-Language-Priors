"""Deep high-level actor-critic policy for Chapter 5 experiments.

The current SAR simulator exposes a team-level discrete action bundle rather
than per-agent continuous controls. This module therefore implements a PPO-style
centralized actor-critic over joint high-level actions. It is a stronger deep-RL
replacement for the tabular learner and can be swapped for full MAPPO once the
simulator exposes step-wise per-agent actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class DeepTrainingTrace:
    episodes: List[int]
    rewards: List[float]
    scores: List[float]
    violations: List[float]
    moving_rewards: List[float]
    moving_scores: List[float]
    moving_violations: List[float]
    losses: List[float]


def moving_average(values: Sequence[float], window: int = 30) -> List[float]:
    result: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result.append(mean(values[start : idx + 1]))
    return result


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(state)
        return self.actor(latent), self.critic(latent).squeeze(-1)


class PPOActorCriticPolicy:
    def __init__(
        self,
        actions: Sequence,
        state_dim: int,
        *,
        device: str | torch.device | None = None,
        lr: float = 3e-4,
        clip_ratio: float = 0.20,
        entropy_coef: float = 0.015,
        value_coef: float = 0.45,
    ) -> None:
        self.actions = list(actions)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCriticNet(state_dim, len(self.actions)).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def distribution(self, states: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, values = self.net(states.to(self.device))
        return Categorical(logits=logits), values

    def sample_action(self, state: Sequence[float]) -> tuple[int, object, float, float]:
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        dist, value = self.distribution(state_tensor)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        idx = int(action_idx.item())
        return idx, self.actions[idx], float(log_prob.item()), float(value.item())

    def greedy_action(self, state: Sequence[float]):
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(state_tensor)
            idx = int(torch.argmax(logits, dim=-1).item())
        return self.actions[idx]

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        *,
        ppo_epochs: int = 4,
    ) -> float:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        losses: List[float] = []
        for _ in range(ppo_epochs):
            dist, values = self.distribution(states_t)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

            ratio = torch.exp(log_probs - old_log_probs_t)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (returns_t - values).pow(2).mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        return mean(losses)


def train_ppo_actor_critic(
    policy: PPOActorCriticPolicy,
    scenario_sampler: Callable,
    prior_builder: Callable,
    evaluator: Callable,
    feature_builder: Callable,
    *,
    episodes: int,
    seed: int,
    planner_enabled: bool,
    safety_enabled: bool,
    batch_size: int = 64,
) -> DeepTrainingTrace:
    rng = Random(seed)
    rewards: List[float] = []
    scores: List[float] = []
    violations: List[float] = []
    losses: List[float] = []

    episode = 0
    while episode < episodes:
        batch_states: List[np.ndarray] = []
        batch_actions: List[int] = []
        batch_old_log_probs: List[float] = []
        batch_returns: List[float] = []

        current_batch = min(batch_size, episodes - episode)
        for _ in range(current_batch):
            scenario = scenario_sampler(episode, rng)
            prior = prior_builder(scenario, planner_enabled)
            features = np.asarray(feature_builder(scenario, prior, planner_enabled), dtype=np.float32)
            action_idx, action, old_log_prob, _ = policy.sample_action(features)
            outcome = evaluator(scenario, action, prior, safety_enabled)

            batch_states.append(features)
            batch_actions.append(action_idx)
            batch_old_log_probs.append(old_log_prob)
            batch_returns.append(float(outcome["reward"]))
            rewards.append(float(outcome["reward"]))
            scores.append(float(outcome["composite_score"]))
            violations.append(float(outcome["violation_rate"]))
            episode += 1

        loss = policy.update(
            np.stack(batch_states, axis=0),
            np.asarray(batch_actions, dtype=np.int64),
            np.asarray(batch_old_log_probs, dtype=np.float32),
            np.asarray(batch_returns, dtype=np.float32),
        )
        losses.append(loss)

    episodes_axis = list(range(1, episodes + 1))
    return DeepTrainingTrace(
        episodes=episodes_axis,
        rewards=rewards,
        scores=scores,
        violations=violations,
        moving_rewards=moving_average(rewards),
        moving_scores=moving_average(scores),
        moving_violations=moving_average(violations),
        losses=losses,
    )


def evaluate_ppo_policy(
    policy: PPOActorCriticPolicy,
    scenarios: Iterable,
    prior_builder: Callable,
    evaluator: Callable,
    feature_builder: Callable,
    *,
    planner_enabled: bool,
    safety_enabled: bool,
) -> List[Dict[str, float]]:
    outcomes: List[Dict[str, float]] = []
    for scenario in scenarios:
        prior = prior_builder(scenario, planner_enabled)
        features = feature_builder(scenario, prior, planner_enabled)
        action = policy.greedy_action(features)
        outcome = evaluator(scenario, action, prior, safety_enabled)
        outcome["policy_type"] = "deep_actor_critic"
        outcomes.append(outcome)
    return outcomes
