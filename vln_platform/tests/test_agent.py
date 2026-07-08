from __future__ import annotations

from pathlib import Path

from src.core.agent import UnifiedNavigationAgent
from src.schemas.data_models import OdometryData


def test_agent_safety_interceptor_on_close_proximity() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "agent_config.yaml"
    agent = UnifiedNavigationAgent(config_path=config_path)
    dangerous_odom = OdometryData(x=10.0, y=10.0, z=0.0, roll=0.0, pitch=0.0, yaw=1.57)

    action = agent.compute_next_action(
        language_instruction="Go to the desk",
        odometry=dangerous_odom,
        mock_proximity_distance=0.15,
    )

    assert action.linear_velocity == 0.0
    assert action.angular_velocity == 0.0
    assert action.is_terminal is False
