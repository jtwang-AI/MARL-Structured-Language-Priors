from dataclasses import dataclass
from typing import Dict, Tuple


Position = Tuple[float, float]


@dataclass(frozen=True)
class Agent:
    agent_id: str
    kind: str
    position: Position
    speed: float
    energy: float
    behavior: Dict[str, float]


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    position: Position
    information_gain: float
    rescue_value: float
    risk_cost: float
    target_probability: float = 0.0
