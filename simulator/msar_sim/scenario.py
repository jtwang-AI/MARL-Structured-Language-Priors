from dataclasses import dataclass
from random import Random
from typing import Dict, List, Tuple

from .model import Agent, Task


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    agents: List[Agent]
    search_tasks: List[Task]
    verify_task: Task
    target_position: Tuple[float, float]
    true_search_task_id: str
    sea_state: float
    target_speed: float
    communication_quality: float
    difficulty: str
    regime: str


def _sample_behavior(rng: Random, kind: str) -> Dict[str, float]:
    if kind == "UAV":
        return {
            "info_pref": rng.uniform(1.1, 1.6),
            "rescue_pref": rng.uniform(0.5, 0.9),
            "risk_aversion": rng.uniform(0.4, 0.8),
            "travel_penalty": rng.uniform(0.2, 0.5),
            "search_efficiency": rng.uniform(1.1, 1.4),
            "verify_efficiency": rng.uniform(0.7, 1.0),
        }
    return {
        "info_pref": rng.uniform(0.6, 1.0),
        "rescue_pref": rng.uniform(1.0, 1.4),
        "risk_aversion": rng.uniform(0.8, 1.2),
        "travel_penalty": rng.uniform(0.6, 1.0),
        "search_efficiency": rng.uniform(0.8, 1.0),
        "verify_efficiency": rng.uniform(1.0, 1.3),
    }


def build_demo_scenario() -> Tuple[List[Agent], List[Task], Tuple[float, float]]:
    scenario = generate_scenario(seed=7, scenario_id=0, num_uavs=1, num_usvs=3, num_search_tasks=4)
    return scenario.agents, scenario.search_tasks + [scenario.verify_task], scenario.target_position


def generate_scenario(
    seed: int,
    scenario_id: int,
    num_uavs: int = 2,
    num_usvs: int = 4,
    num_search_tasks: int = 6,
    regime: str = "standard",
) -> Scenario:
    rng = Random(seed)
    if regime == "stress":
        sea_state = rng.uniform(3.6, 5.5)
        target_speed = rng.uniform(1.8, 3.4)
        communication_quality = rng.uniform(0.20, 0.55)
    else:
        sea_state = rng.uniform(1.0, 4.0)
        target_speed = rng.uniform(0.4, 2.2)
        communication_quality = rng.uniform(0.55, 0.98)
    target_position = (rng.uniform(30.0, 80.0), rng.uniform(-20.0, 30.0))

    agents: List[Agent] = []
    for idx in range(num_uavs):
        agents.append(
            Agent(
                agent_id=f"UAV_{idx + 1}",
                kind="UAV",
                position=(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0)),
                speed=rng.uniform(16.0, 22.0),
                energy=rng.uniform(0.75, 0.95),
                behavior=_sample_behavior(rng, "UAV"),
            )
        )
    for idx in range(num_usvs):
        agents.append(
            Agent(
                agent_id=f"USV_{idx + 1}",
                kind="USV",
                position=(rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)),
                speed=rng.uniform(4.5, 7.5),
                energy=rng.uniform(0.70, 0.92),
                behavior=_sample_behavior(rng, "USV"),
            )
        )

    true_search_idx = rng.randrange(num_search_tasks)
    search_tasks: List[Task] = []
    for idx in range(num_search_tasks):
        if idx == true_search_idx:
            position = (
                target_position[0] + rng.uniform(-8.0, 8.0),
                target_position[1] + rng.uniform(-8.0, 8.0),
            )
            info_gain = rng.uniform(8.5, 10.0)
            rescue_value = rng.uniform(6.5, 8.5)
            risk_cost = rng.uniform(2.0, 4.0)
            target_probability = rng.uniform(0.72, 0.92)
        else:
            position = (rng.uniform(10.0, 90.0), rng.uniform(-30.0, 35.0))
            info_gain = rng.uniform(4.0, 8.0)
            rescue_value = rng.uniform(2.5, 6.5)
            risk_cost = rng.uniform(1.0, 5.0)
            target_probability = rng.uniform(0.08, 0.42)

        search_tasks.append(
            Task(
                task_id=f"SEARCH_{idx + 1}",
                task_type="search",
                position=position,
                information_gain=info_gain,
                rescue_value=rescue_value,
                risk_cost=risk_cost,
                target_probability=target_probability,
            )
        )

    verify_task = Task(
        task_id="VERIFY_TARGET",
        task_type="verify",
        position=target_position,
        information_gain=5.0,
        rescue_value=10.0,
        risk_cost=2.0 + 0.5 * sea_state,
        target_probability=1.0,
    )

    difficulty_score = 0.45 * sea_state + 0.35 * target_speed + 0.20 * (1.0 - communication_quality) * 5.0
    if difficulty_score < 1.9:
        difficulty = "easy"
    elif difficulty_score < 2.8:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return Scenario(
        scenario_id=scenario_id,
        agents=agents,
        search_tasks=search_tasks,
        verify_task=verify_task,
        target_position=target_position,
        true_search_task_id=search_tasks[true_search_idx].task_id,
        sea_state=sea_state,
        target_speed=target_speed,
        communication_quality=communication_quality,
        difficulty=difficulty,
        regime=regime,
    )
