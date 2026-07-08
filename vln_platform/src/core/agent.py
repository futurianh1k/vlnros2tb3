from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.accelerators.quantization import QuantizationConfig, QuantizedInferenceBackend
from src.core.buffer import StreamBufferSystem
from src.core.scene_graph import SpatialSceneGraphBuilder
from src.schemas.data_models import ActionOutput, OdometryData
from src.utils.safety import SafetyInterceptor, enforce_safety


@dataclass(slots=True)
class AgentConfig:
    model_name: str = "open-source-vlm"
    max_inference_latency_ms: int = 200
    proximity_stop_threshold_m: float = 0.3
    buffer_window_size: int = 5
    emergency_linear_velocity: float = 0.0
    emergency_angular_velocity: float = 0.0


class UnifiedNavigationAgent:
    def __init__(
        self,
        config_path: str | Path,
        buffer_system: Optional[StreamBufferSystem] = None,
        scene_graph: Optional[SpatialSceneGraphBuilder] = None,
        inference_backend: Optional[QuantizedInferenceBackend] = None,
        safety_interceptor: Optional[SafetyInterceptor] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.buffer_system = buffer_system or StreamBufferSystem(max_len=self.config.buffer_window_size)
        self.scene_graph = scene_graph or SpatialSceneGraphBuilder()
        self.inference_backend = inference_backend or QuantizedInferenceBackend(
            QuantizationConfig(model_name=self.config.model_name)
        )
        self.safety_interceptor = safety_interceptor or SafetyInterceptor(
            minimum_clearance_m=self.config.proximity_stop_threshold_m,
            emergency_linear_velocity=self.config.emergency_linear_velocity,
            emergency_angular_velocity=self.config.emergency_angular_velocity,
        )
        self._last_odometry: Optional[OdometryData] = None

    @staticmethod
    def _load_config(config_path: Path) -> AgentConfig:
        if not config_path.exists():
            return AgentConfig()

        raw_text = config_path.read_text(encoding="utf-8")
        if yaml is not None:
            payload = yaml.safe_load(raw_text) or {}
            return AgentConfig(
                model_name=payload.get("model_name", "open-source-vlm"),
                max_inference_latency_ms=int(payload.get("max_inference_latency_ms", 200)),
                proximity_stop_threshold_m=float(payload.get("proximity_stop_threshold_m", 0.3)),
                buffer_window_size=int(payload.get("buffer_window_size", 5)),
                emergency_linear_velocity=float(payload.get("emergency_linear_velocity", 0.0)),
                emergency_angular_velocity=float(payload.get("emergency_angular_velocity", 0.0)),
            )

        payload: dict[str, str] = {}
        current_section: Optional[str] = None
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                current_section = line[:-1]
                continue
            if ":" not in line:
                continue
            key, value = [segment.strip() for segment in line.split(":", 1)]
            if current_section:
                payload[f"{current_section}.{key}"] = value
            else:
                payload[key] = value

        return AgentConfig(
            model_name=payload.get("model_name", "open-source-vlm"),
            max_inference_latency_ms=int(payload.get("max_inference_latency_ms", 200)),
            proximity_stop_threshold_m=float(payload.get("proximity_stop_threshold_m", 0.3)),
            buffer_window_size=int(payload.get("buffer_window_size", 5)),
            emergency_linear_velocity=float(payload.get("emergency_linear_velocity", 0.0)),
            emergency_angular_velocity=float(payload.get("emergency_angular_velocity", 0.0)),
        )

    @staticmethod
    def _tokenize_instruction(language_instruction: str) -> list[str]:
        return [token for token in language_instruction.strip().split() if token]

    @staticmethod
    def _decode_action(raw_action: dict[str, Any]) -> ActionOutput:
        return ActionOutput(**raw_action)

    def ingest_odometry(self, odometry: OdometryData) -> None:
        self._last_odometry = odometry

    def ingest_frame(self, frame: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        self.buffer_system.push(frame, metadata=metadata)

    def compute_next_action(
        self,
        language_instruction: str,
        odometry: Optional[OdometryData] = None,
        mock_proximity_distance: Optional[float] = None,
        mock_inference_delay_ms: Optional[int] = None,
    ) -> ActionOutput:
        active_odometry = odometry or self._last_odometry
        buffered_frames = self.buffer_system.latest_batch(batch_size=self.config.buffer_window_size)
        tokens = self._tokenize_instruction(language_instruction)

        start_time = perf_counter()
        if mock_inference_delay_ms is not None and mock_inference_delay_ms > 0:
            from time import sleep

            sleep(mock_inference_delay_ms / 1000.0)

        try:
            raw_action = self.inference_backend.forward(tokens=tokens, frames=buffered_frames, odometry=active_odometry)
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            if elapsed_ms > self.config.max_inference_latency_ms:
                return self.safety_interceptor.emergency_stop()

            decoded_action = self._decode_action(raw_action)
            return enforce_safety(decoded_action, mock_proximity_distance, self.safety_interceptor)
        except Exception:
            return self.safety_interceptor.emergency_stop()


__all__ = ["AgentConfig", "UnifiedNavigationAgent"]
