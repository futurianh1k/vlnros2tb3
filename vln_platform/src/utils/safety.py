from __future__ import annotations

from src.schemas.data_models import ActionOutput


def should_emergency_brake(proximity_distance_m: float | None, threshold_m: float = 0.3) -> bool:
    return proximity_distance_m is not None and proximity_distance_m < threshold_m


def apply_safety_interceptor(
    action: ActionOutput,
    proximity_distance_m: float | None,
    threshold_m: float = 0.3,
) -> ActionOutput:
    if should_emergency_brake(proximity_distance_m, threshold_m):
        return ActionOutput(
            linear_velocity=0.0,
            angular_velocity=0.0,
            waypoint=action.waypoint,
            is_terminal=False,
        )
    return action

