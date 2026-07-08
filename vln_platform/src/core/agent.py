from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.accelerators.quantization import run_fp8_forward_pass
from src.core.buffer import StreamBufferSystem
from src.schemas.data_models import ActionOutput, OdometryData
from src.utils.safety import apply_safety_interceptor

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_buffer_len": 32,
    "inference_timeout_ms": 200,
    "default_linear_velocity": 0.2,
    "default_angular_velocity": 0.0,
    "proximity_stop_threshold_m": 0.3,
}


class UnifiedNavigationAgent:
    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(config_path)
        self.buffer = StreamBufferSystem(max_len=int(self.config["max_buffer_len"]))

    def compute_next_action(
        self,
        language_instruction: str,
        odometry_data: Optional[OdometryData] = None,
        mock_proximity_distance: Optional[float] = None,
    ) -> ActionOutput:
        start_time = time.perf_counter()

        frame_context = self.buffer.pop_all_context(latest_n=5)
        _ = odometry_data

        text_embedding = self._tokenize_instruction(language_instruction)
        model_output = run_fp8_forward_pass(frame_context, text_embedding)
        action = self._decode_action(model_output)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > float(self.config["inference_timeout_ms"]):
            return ActionOutput(linear_velocity=0.0, angular_velocity=0.0, waypoint=None, is_terminal=False)

        return apply_safety_interceptor(
            action=action,
            proximity_distance_m=mock_proximity_distance,
            threshold_m=float(self.config["proximity_stop_threshold_m"]),
        )

    @staticmethod
    def _tokenize_instruction(language_instruction: str) -> np.ndarray:
        token_values = [float(ord(ch) % 256) / 255.0 for ch in language_instruction]
        return np.asarray(token_values, dtype=np.float32)

    def _decode_action(self, logits: np.ndarray) -> ActionOutput:
        linear_velocity = float(np.clip(self.config["default_linear_velocity"] + logits[0], -0.5, 1.5))
        angular_velocity = float(np.clip(self.config["default_angular_velocity"] + logits[1], -1.0, 1.0))
        return ActionOutput(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            waypoint=[0.0, 0.0],
            is_terminal=False,
        )

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return dict(DEFAULT_CONFIG)

        loaded: Dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                loaded[key.strip()] = _coerce_yaml_scalar(value.strip())

        merged = dict(DEFAULT_CONFIG)
        merged.update(loaded)
        return merged


def _coerce_yaml_scalar(value: str) -> Any:
    lowered = value.lower().strip("\"'")
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value.strip("\"'")

