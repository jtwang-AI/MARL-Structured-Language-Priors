from typing import Dict, Iterable, List

from .model import Agent, Task
from .utility import utility


def rank_tasks(agent: Agent, tasks: Iterable[Task]) -> List[Task]:
    return sorted(tasks, key=lambda task: utility(agent, task), reverse=True)


def distributed_greedy_assignment(agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
    remaining_tasks = {task.task_id: task for task in tasks}
    assignment: Dict[str, str] = {}

    # Simplified distributed bidding surrogate:
    # each agent bids for its best remaining task, highest utility wins.
    while remaining_tasks:
        bids = []
        for agent in agents:
            if agent.agent_id in assignment:
                continue
            ranked = rank_tasks(agent, remaining_tasks.values())
            if ranked:
                best = ranked[0]
                bids.append((utility(agent, best), agent.agent_id, best.task_id))

        if not bids:
            break

        bids.sort(reverse=True)
        _, winner_agent, winner_task = bids[0]
        assignment[winner_agent] = winner_task
        remaining_tasks.pop(winner_task, None)

    return assignment
