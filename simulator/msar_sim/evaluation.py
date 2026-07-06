from dataclasses import dataclass
from math import exp
from statistics import mean
from typing import Dict, List, Optional

from .baselines import (
    behavior_distributed,
    centralized_optimal_assignment,
    distance_only_assignment,
    homogeneous_distributed,
)
from .formation import adaptive_radius, circular_formation_targets
from .model import Agent, Task
from .scenario import Scenario
from .utility import distance, travel_cost, utility


METHODS = {
    "behavior_distributed": behavior_distributed,
    "homogeneous_distributed": homogeneous_distributed,
    "distance_only": distance_only_assignment,
    "centralized_optimal": centralized_optimal_assignment,
}


ABLATIONS = {
    "full": {
        "use_behavior": True,
        "use_risk": True,
        "use_rescue": True,
        "use_mode_switch": True,
        "adaptive_radius": True,
    },
    "no_risk": {
        "use_behavior": True,
        "use_risk": False,
        "use_rescue": True,
        "use_mode_switch": True,
        "adaptive_radius": True,
    },
    "no_rescue_value": {
        "use_behavior": True,
        "use_risk": True,
        "use_rescue": False,
        "use_mode_switch": True,
        "adaptive_radius": True,
    },
    "no_mode_switch": {
        "use_behavior": True,
        "use_risk": True,
        "use_rescue": True,
        "use_mode_switch": False,
        "adaptive_radius": True,
    },
    "fixed_radius": {
        "use_behavior": True,
        "use_risk": True,
        "use_rescue": True,
        "use_mode_switch": True,
        "adaptive_radius": False,
    },
}


@dataclass(frozen=True)
class ExperimentResult:
    scenario_id: int
    difficulty: str
    regime: str
    method: str
    detection_time: float
    verification_time: float
    encirclement_time: float
    mission_time: float
    total_path_length: float
    utility_score: float
    formation_error: float
    rescue_success: float
    communication_load: float
    communication_quality: float
    sea_state: float
    target_speed: float


def _search_efficiency(agent: Agent) -> float:
    return agent.behavior.get("search_efficiency", 1.0)


def _verify_efficiency(agent: Agent) -> float:
    return agent.behavior.get("verify_efficiency", 1.0)


def _assigned_task(tasks: List[Task], task_id: str) -> Task:
    lookup = {task.task_id: task for task in tasks}
    return lookup[task_id]


def _task_with_ablation(task: Task, config: Dict[str, bool]) -> Task:
    return Task(
        task_id=task.task_id,
        task_type=task.task_type,
        position=task.position,
        information_gain=task.information_gain,
        rescue_value=task.rescue_value if config["use_rescue"] else 0.0,
        risk_cost=task.risk_cost if config["use_risk"] else 0.0,
        target_probability=task.target_probability,
    )


def _agent_with_ablation(agent: Agent, config: Dict[str, bool]) -> Agent:
    if config["use_behavior"]:
        return agent
    flat_behavior = {key: 1.0 for key in agent.behavior}
    return Agent(
        agent_id=agent.agent_id,
        kind=agent.kind,
        position=agent.position,
        speed=agent.speed,
        energy=agent.energy,
        behavior=flat_behavior,
    )


def _detection_time(scenario: Scenario, assignment: Dict[str, str]) -> float:
    true_task = _assigned_task(scenario.search_tasks, scenario.true_search_task_id)
    assigned_agents = [
        agent
        for agent in scenario.agents
        if assignment.get(agent.agent_id) == scenario.true_search_task_id
    ]
    if not assigned_agents:
        closest = min(
            scenario.agents,
            key=lambda agent: distance(agent.position, true_task.position),
        )
        fallback = travel_cost(closest, true_task)
        penalty = 2.1 if scenario.regime == "stress" else 1.8
        return penalty * fallback / max(_search_efficiency(closest) * true_task.target_probability, 1e-6)

    base_time = min(
        travel_cost(agent, true_task) / max(_search_efficiency(agent) * true_task.target_probability, 1e-6)
        for agent in assigned_agents
    )
    if scenario.regime == "stress":
        base_time *= 1.0 + 0.12 * scenario.sea_state
    return base_time


def _verification_time(
    scenario: Scenario,
    detection_time: float,
    method: str,
    verify_task: Task,
    agents: List[Agent],
    config: Dict[str, bool],
) -> float:
    verify_candidates = [agent for agent in agents if agent.kind in {"UAV", "USV"}]
    if method == "distance_only":
        verifier = min(
            verify_candidates,
            key=lambda agent: distance(agent.position, verify_task.position),
        )
    else:
        verifier = max(verify_candidates, key=lambda agent: utility(agent, verify_task))
    verify_leg = travel_cost(verifier, verify_task) / max(_verify_efficiency(verifier), 1e-6)
    communication_penalty = 1.0 - scenario.communication_quality
    if method == "centralized_optimal":
        verify_leg *= 1.0 + 0.55 * communication_penalty
    elif method in {"behavior_distributed", "homogeneous_distributed"}:
        verify_leg *= 1.0 + 0.20 * communication_penalty
    else:
        verify_leg *= 1.0 + 0.10 * communication_penalty
    if not config["use_mode_switch"]:
        verify_leg *= 1.15
    return detection_time + verify_leg


def _encirclement_metrics(
    scenario: Scenario,
    method: str,
    verification_time: float,
    config: Dict[str, bool],
    agents: List[Agent],
) -> Dict[str, float]:
    usvs = [agent for agent in agents if agent.kind == "USV"]
    if method == "distance_only" or not config["adaptive_radius"]:
        radius = 25.0
    else:
        radius = adaptive_radius(25.0, scenario.target_speed, scenario.sea_state)

    slots = circular_formation_targets(
        center=scenario.target_position,
        num_agents=len(usvs),
        base_radius=25.0,
        target_speed=0.0 if method == "distance_only" or not config["adaptive_radius"] else scenario.target_speed,
        sea_state=0.0 if method == "distance_only" or not config["adaptive_radius"] else scenario.sea_state,
    )
    slot_tasks = [
        Task(
            task_id=f"SLOT_{idx + 1}",
            task_type="encircle",
            position=slot,
            information_gain=0.0,
            rescue_value=8.0 if config["use_rescue"] else 0.0,
            risk_cost=(1.5 + 0.2 * scenario.sea_state) if config["use_risk"] else 0.0,
            target_probability=1.0,
        )
        for idx, slot in enumerate(slots)
    ]
    if method == "distance_only":
        slot_assignment = distance_only_assignment(usvs, slot_tasks)
        communication_load = 1.0 * len(usvs)
    elif method == "centralized_optimal":
        slot_assignment = centralized_optimal_assignment(usvs, slot_tasks)
        communication_load = 3.0 * len(usvs)
    elif method == "homogeneous_distributed":
        slot_assignment = homogeneous_distributed(usvs, slot_tasks)
        communication_load = 1.6 * len(usvs)
    else:
        slot_assignment = behavior_distributed(usvs, slot_tasks)
        communication_load = 1.8 * len(usvs)

    slot_map = {task.task_id: task for task in slot_tasks}
    travel_times = []
    tracking_errors = []
    path_lengths = []
    packet_loss_penalty = 1.0 - scenario.communication_quality
    if method == "centralized_optimal":
        coordination_penalty = 0.55 * packet_loss_penalty
    elif method in {"behavior_distributed", "homogeneous_distributed"}:
        coordination_penalty = 0.22 * packet_loss_penalty
    else:
        coordination_penalty = 0.28 * packet_loss_penalty
    for agent in usvs:
        task_id = slot_assignment.get(agent.agent_id)
        if task_id is None:
            continue
        slot_task = slot_map[task_id]
        leg = travel_cost(agent, slot_task)
        travel_times.append(leg)
        slot_distance = distance(agent.position, slot_task.position)
        path_lengths.append(slot_distance)
        tracking_errors.append(
            0.055 * slot_distance
            + coordination_penalty
            + 0.08 * scenario.sea_state
            + 0.06 * max(scenario.target_speed - 0.8, 0.0)
        )

    encirclement_time = verification_time + (max(travel_times) if travel_times else 0.0)
    formation_error = mean(tracking_errors) if tracking_errors else 0.0
    total_path = sum(path_lengths)
    if method == "centralized_optimal":
        encirclement_time *= 1.0 + 0.65 * packet_loss_penalty
    elif method in {"behavior_distributed", "homogeneous_distributed"}:
        encirclement_time *= 1.0 + 0.25 * packet_loss_penalty
    else:
        encirclement_time *= 1.0 + 0.12 * packet_loss_penalty
    if scenario.regime == "stress":
        encirclement_time *= 1.0 + 0.08 * scenario.sea_state
        formation_error += 0.4 * max(scenario.target_speed - 1.8, 0.0)
    if not config["use_mode_switch"]:
        encirclement_time *= 1.10
    if not config["adaptive_radius"]:
        encirclement_time *= 1.0 + 0.05 * max(scenario.target_speed - 1.0, 0.0) + 0.04 * max(scenario.sea_state - 2.0, 0.0)
        formation_error += 1.1 * max(scenario.target_speed - 1.0, 0.0) + 0.8 * max(scenario.sea_state - 2.0, 0.0)
    rescue_success = exp(
        -0.035 * encirclement_time
        - 0.08 * formation_error
        - 0.04 * scenario.sea_state
        - 0.02 * max(radius - 25.0, 0.0)
    )
    return {
        "encirclement_time": encirclement_time,
        "formation_error": formation_error,
        "total_path": total_path,
        "rescue_success": rescue_success,
        "communication_load": communication_load,
    }


def evaluate_method(
    scenario: Scenario,
    method: str,
    ablation: Optional[str] = None,
) -> ExperimentResult:
    config = ABLATIONS[ablation or "full"]
    assigner = METHODS[method]
    agents = [_agent_with_ablation(agent, config) for agent in scenario.agents]
    search_tasks = [_task_with_ablation(task, config) for task in scenario.search_tasks]
    verify_task = _task_with_ablation(scenario.verify_task, config)

    eval_scenario = Scenario(
        scenario_id=scenario.scenario_id,
        agents=agents,
        search_tasks=search_tasks,
        verify_task=verify_task,
        target_position=scenario.target_position,
        true_search_task_id=scenario.true_search_task_id,
        sea_state=scenario.sea_state,
        target_speed=scenario.target_speed,
        communication_quality=scenario.communication_quality,
        difficulty=scenario.difficulty,
        regime=scenario.regime,
    )

    search_assignment = assigner(agents, search_tasks)
    detection_time = _detection_time(eval_scenario, search_assignment)
    verification_time = _verification_time(eval_scenario, detection_time, method, verify_task, agents, config)
    encirclement = _encirclement_metrics(eval_scenario, method, verification_time, config, agents)

    search_utility = 0.0
    search_path = 0.0
    task_map = {task.task_id: task for task in search_tasks}
    for agent in agents:
        task_id = search_assignment.get(agent.agent_id)
        if task_id is None:
            continue
        task = task_map[task_id]
        search_utility += utility(agent, task)
        search_path += distance(agent.position, task.position)

    mission_time = encirclement["encirclement_time"]
    total_path_length = search_path + encirclement["total_path"]
    utility_score = search_utility + 10.0 * encirclement["rescue_success"]

    return ExperimentResult(
        scenario_id=scenario.scenario_id,
        difficulty=eval_scenario.difficulty,
        regime=eval_scenario.regime,
        method=method if ablation is None else f"{method}__{ablation}",
        detection_time=detection_time,
        verification_time=verification_time,
        encirclement_time=encirclement["encirclement_time"],
        mission_time=mission_time,
        total_path_length=total_path_length,
        utility_score=utility_score,
        formation_error=encirclement["formation_error"],
        rescue_success=encirclement["rescue_success"],
        communication_load=encirclement["communication_load"],
        communication_quality=eval_scenario.communication_quality,
        sea_state=eval_scenario.sea_state,
        target_speed=eval_scenario.target_speed,
    )
