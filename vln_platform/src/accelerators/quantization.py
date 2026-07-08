from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class QuantizationConfig:
    model_name: str = "open-source-vlm"
    precision: str = "fp8"
    backend: str = "vllm"
    enabled: bool = True


@dataclass(slots=True)
class QuantizedInferenceBackend:
    config: QuantizationConfig

    def forward(
        self,
        tokens: list[str],
        frames: Optional[Any] = None,
        odometry: Optional[Any] = None,
    ) -> dict[str, Any]:
        instruction = " ".join(tokens).lower()

        linear_velocity = 0.0
        angular_velocity = 0.0

        if any(keyword in instruction for keyword in ("left", "turn left")):
            angular_velocity = 0.35
        elif any(keyword in instruction for keyword in ("right", "turn right")):
            angular_velocity = -0.35

        if any(keyword in instruction for keyword in ("go", "move", "forward", "approach")):
            linear_velocity = 0.25

        if "desk" in instruction or "table" in instruction:
            linear_velocity = max(linear_velocity, 0.3)

        if frames is not None and getattr(frames, "shape", None) is not None and frames.shape[0] >= 5:
            linear_velocity = min(0.5, linear_velocity + 0.1)

        waypoint = [0.0, 0.0] if linear_velocity > 0.0 else None

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "waypoint": waypoint,
            "is_terminal": False,
        }


__all__ = ["QuantizationConfig", "QuantizedInferenceBackend"]
