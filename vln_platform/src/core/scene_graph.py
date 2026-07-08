from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import networkx as nx

from src.schemas.data_models import SpatialNode


def _euclidean_distance(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    return sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_values, right_values)))


@dataclass(slots=True)
class SceneGraphUpdateResult:
    node_id: str
    merged: bool


class SpatialSceneGraphBuilder:
    def __init__(self, merge_distance_m: float = 1.5) -> None:
        self.graph = nx.Graph()
        self.merge_distance_m = merge_distance_m

    def _candidate_nodes(self, node: SpatialNode) -> list[tuple[str, dict]]:
        candidates: list[tuple[str, dict]] = []
        for node_id, attributes in self.graph.nodes(data=True):
            if attributes.get("label") != node.label:
                continue
            distance = _euclidean_distance(attributes["coordinates_3d"], node.coordinates_3d)
            if distance <= self.merge_distance_m:
                candidates.append((node_id, attributes))
        return candidates

    def add_or_update_node(self, node: SpatialNode) -> SceneGraphUpdateResult:
        candidates = self._candidate_nodes(node)
        if not candidates:
            self.graph.add_node(
                node.node_id,
                label=node.label,
                confidence=node.confidence,
                coordinates_3d=list(node.coordinates_3d),
                observations=1,
            )
            return SceneGraphUpdateResult(node_id=node.node_id, merged=False)

        candidate_id, candidate_attributes = min(
            candidates,
            key=lambda item: _euclidean_distance(item[1]["coordinates_3d"], node.coordinates_3d),
        )

        previous_coordinates = candidate_attributes["coordinates_3d"]
        previous_confidence = float(candidate_attributes.get("confidence", 0.5))
        blend_factor = max(0.1, min(0.9, node.confidence))
        updated_coordinates = [
            (1.0 - blend_factor) * previous + blend_factor * current
            for previous, current in zip(previous_coordinates, node.coordinates_3d)
        ]
        updated_confidence = min(1.0, 0.7 * previous_confidence + 0.3 * node.confidence)
        observations = int(candidate_attributes.get("observations", 1)) + 1

        self.graph.nodes[candidate_id].update(
            {
                "label": node.label,
                "confidence": updated_confidence,
                "coordinates_3d": updated_coordinates,
                "observations": observations,
            }
        )
        return SceneGraphUpdateResult(node_id=candidate_id, merged=True)

    def get_graph(self) -> nx.Graph:
        return self.graph


__all__ = ["SceneGraphUpdateResult", "SpatialSceneGraphBuilder"]
