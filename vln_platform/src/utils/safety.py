from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SafetyInterceptor:
    minimum_clearance_m: float = 0.3
    emergency_linear_velocity: float = 0.0
    emergency_angular_velocity: float = 0.0

    def should_stop(self, proximity_distance_m: Optional[float]) -> bool:
        return proximity_distance_m is not None and proximity_distance_m < self.minimum_clearance_m

    def emergency_stop(self, waypoint: Optional[list[float]] = None):
        from src.schemas.data_models import ActionOutput

        return ActionOutput(
            linear_velocity=self.emergency_linear_velocity,
            angular_velocity=self.emergency_angular_velocity,
            waypoint=waypoint,
            is_terminal=False,
        )


def enforce_safety(action_output, proximity_distance_m: Optional[float], interceptor: Optional[SafetyInterceptor] = None):
    active_interceptor = interceptor or SafetyInterceptor()
    if active_interceptor.should_stop(proximity_distance_m):
        return active_interceptor.emergency_stop(getattr(action_output, "waypoint", None))
    return action_output


__all__ = ["SafetyInterceptor", "enforce_safety"]
