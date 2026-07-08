from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.agent import UnifiedNavigationAgent
from src.schemas.data_models import OdometryData


def test_agent_safety_interceptor_on_close_proximity():
    agent = UnifiedNavigationAgent(config_path="config/agent_config.yaml")
    dangerous_odom = OdometryData(x=10.0, y=10.0, z=0.0, roll=0.0, pitch=0.0, yaw=1.57)

    action = agent.compute_next_action(
        language_instruction="Go to the desk",
        odometry_data=dangerous_odom,
        mock_proximity_distance=0.15,
    )

    assert action.linear_velocity == 0.0
    assert action.angular_velocity == 0.0
    assert action.is_terminal is False
