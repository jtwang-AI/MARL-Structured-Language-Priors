from math import cos, pi, sin
from typing import List, Tuple


Position = Tuple[float, float]


def adaptive_radius(base_radius: float, target_speed: float, sea_state: float) -> float:
    return base_radius + 3.0 * target_speed + 1.5 * sea_state


def circular_formation_targets(
    center: Position,
    num_agents: int,
    base_radius: float,
    target_speed: float,
    sea_state: float,
) -> List[Position]:
    radius = adaptive_radius(base_radius, target_speed, sea_state)
    slots: List[Position] = []
    for idx in range(num_agents):
        theta = 2.0 * pi * idx / max(num_agents, 1)
        slots.append(
            (
                center[0] + radius * cos(theta),
                center[1] + radius * sin(theta),
            )
        )
    return slots
