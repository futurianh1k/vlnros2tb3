from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OdometryData(BaseModel):
    x: float = Field(..., description="로봇 중심의 X 좌표 (미터 단위)")
    y: float = Field(..., description="로봇 중심의 Y 좌표 (미터 단위)")
    z: float = Field(..., description="로봇 중심의 Z 좌표 (미터 단위)")
    roll: float = Field(..., description="오리엔테이션 Roll (라디안)")
    pitch: float = Field(..., description="오리엔테이션 Pitch (라디안)")
    yaw: float = Field(..., description="오리엔테이션 Yaw (라디안)")


class SpatialNode(BaseModel):
    node_id: str
    label: str = Field(..., description="오브젝트 시맨틱 클래스 (예: 'chair', 'desk', 'door')")
    confidence: float = Field(..., description="VLM 인지 신뢰도 점수 (0.0 ~ 1.0)")
    coordinates_3d: List[float] = Field(
        ...,
        alias="3d_coordinates",
        min_length=3,
        max_length=3,
        description="[X, Y, Z] 절대 위치 정보",
    )


class ActionOutput(BaseModel):
    linear_velocity: float = Field(..., description="선속도 제어값 (m/s, 범위: -0.5 ~ 1.5)")
    angular_velocity: float = Field(..., description="각속도 제어값 (rad/s, 범위: -1.0 ~ 1.0)")
    waypoint: Optional[List[float]] = Field(None, description="단기 목표 로컬 타깃 격자 좌표 [X, Y]")
    is_terminal: bool = Field(False, description="목적지 도달 및 태스크 완수 여부 신호")
