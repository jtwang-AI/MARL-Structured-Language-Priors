"""Tabular high-level policy learner for Chapter 5."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Dict, List, Tuple


@dataclass
class TrainingTrace:
    episodes: List[int]
    rewards: List[float]
    scores: List[float]
    violations: List[float]
    moving_rewards: List[float]
    moving_scores: List[float]
    moving_violations: List[float]


class TabularPolicy:
    def __init__(self, actions: List, alpha: float = 0.18, gamma: float = 0.15):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: {action.label: 0.0 for action in actions})

    def greedy_action(self, state: str):
        values = self.q_table[state]
        best_label = max(values, key=values.get)
        return next(action for action in self.actions if action.label == best_label)

    def choose_action(self, state: str, epsilon: float, rng: Random):
        if rng.random() < epsilon:
            return rng.choice(self.actions)
        return self.greedy_action(state)

    def update(self, state: str, action_label: str, reward: float, next_state: str) -> None:
        current = self.q_table[state][action_label]
        future = max(self.q_table[next_state].values())
        self.q_table[state][action_label] = current + self.alpha * (reward + self.gamma * future - current)


def moving_average(values: List[float], window: int = 20) -> List[float]:
    result: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result.append(mean(values[start : idx + 1]))
    return result


def train_policy(
    policy: TabularPolicy,
    scenario_sampler,
    prior_builder,
    evaluator,
    episodes: int,
    seed: int,
    planner_enabled: bool,
    safety_enabled: bool,
) -> TrainingTrace:
    rng = Random(seed)
    rewards: List[float] = []
    scores: List[float] = []
    violations: List[float] = []

    scenario = scenario_sampler(0, rng)
    prior = prior_builder(scenario, planner_enabled)
    state = prior.state_token

    for episode in range(episodes):
        epsilon = max(0.06, 0.30 * (1.0 - episode / max(episodes - 1, 1)))
        action = policy.choose_action(state, epsilon, rng)
        outcome = evaluator(scenario, action, prior, safety_enabled)
        next_scenario = scenario_sampler(episode + 1, rng)
        next_prior = prior_builder(next_scenario, planner_enabled)
        next_state = next_prior.state_token
        policy.update(state, action.label, outcome["reward"], next_state)
        rewards.append(outcome["reward"])
        scores.append(outcome["composite_score"])
        violations.append(outcome["violation_rate"])
        scenario = next_scenario
        prior = next_prior
        state = next_state

    episodes_axis = list(range(1, episodes + 1))
    return TrainingTrace(
        episodes=episodes_axis,
        rewards=rewards,
        scores=scores,
        violations=violations,
        moving_rewards=moving_average(rewards),
        moving_scores=moving_average(scores),
        moving_violations=moving_average(violations),
    )


def evaluate_policy(policy: TabularPolicy, scenarios: List, prior_builder, evaluator, planner_enabled: bool, safety_enabled: bool) -> List[Dict[str, float]]:
    outcomes: List[Dict[str, float]] = []
    for scenario in scenarios:
        prior = prior_builder(scenario, planner_enabled)
        state = prior.state_token
        action = policy.greedy_action(state)
        outcome = evaluator(scenario, action, prior, safety_enabled)
        outcome["policy_type"] = "learned"
        outcomes.append(outcome)
    return outcomes
