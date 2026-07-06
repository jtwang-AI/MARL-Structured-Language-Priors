from copy import deepcopy
from itertools import combinations, permutations
from typing import Callable, Dict, List

from .assignment import distributed_greedy_assignment
from .model import Agent, Task
from .utility import distance, utility


AssignmentMethod = Callable[[List[Agent], List[Task]], Dict[str, str]]


def behavior_distributed(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    return distributed_greedy_assignment(agents, tasks)


def homogeneous_distributed(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    cloned = deepcopy(agents)
    avg = {
        key: sum(agent.behavior.get(key, 0.0) for agent in agents) / len(agents)
        for key in agents[0].behavior
    }
    for agent in cloned:
        agent.behavior.update(avg)
    return distributed_greedy_assignment(cloned, tasks)


def distance_only_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}
    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not remaining_tasks:
            break
        best_task = min(
            remaining_tasks.values(),
            key=lambda task: distance(agent.position, task.position),
        )
        assignment[agent.agent_id] = best_task.task_id
        remaining_tasks.pop(best_task.task_id, None)
    return assignment


def centralized_optimal_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    best_score = None
    best_assignment: Dict[str, str] = {}
    pick_count = min(len(tasks), len(agents))
    for task_subset in combinations(tasks, pick_count):
        for perm in permutations(task_subset, pick_count):
            score = 0.0
            assignment: Dict[str, str] = {}
            for agent, task in zip(agents, perm):
                score += utility(agent, task)
                assignment[agent.agent_id] = task.task_id
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = assignment
    return best_assignment
