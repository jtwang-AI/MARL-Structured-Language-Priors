from math import sqrt

from .model import Agent, Position, Task


def distance(a: Position, b: Position) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def travel_cost(agent: Agent, task: Task) -> float:
    return distance(agent.position, task.position) / max(agent.speed, 1e-6)


def utility(agent: Agent, task: Task) -> float:
    alpha_info = agent.behavior.get("info_pref", 1.0)
    alpha_rescue = agent.behavior.get("rescue_pref", 1.0)
    alpha_risk = agent.behavior.get("risk_aversion", 1.0)
    alpha_travel = agent.behavior.get("travel_penalty", 1.0)

    return (
        alpha_info * task.information_gain
        + alpha_rescue * task.rescue_value
        - alpha_risk * task.risk_cost
        - alpha_travel * travel_cost(agent, task)
    )
