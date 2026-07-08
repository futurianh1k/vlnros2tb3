from __future__ import annotations

from math import sqrt
from typing import Dict, Optional

import networkx as nx

from src.schemas.data_models import SpatialNode


class SpatialSceneGraphBuilder:
    def __init__(self, merge_distance_m: float = 1.5, momentum: float = 0.6) -> None:
        self.graph = nx.Graph()
        self.merge_distance_m = merge_distance_m
        self.momentum = momentum

    def add_or_update_node(self, node: SpatialNode) -> str:
        matched_id = self._find_merge_candidate(node)
        if matched_id is None:
            self.graph.add_node(
                node.node_id,
                label=node.label,
                confidence=node.confidence,
                coordinates_3d=node.coordinates_3d,
            )
            return node.node_id

        attrs: Dict[str, object] = self.graph.nodes[matched_id]
        prev_conf = float(attrs.get("confidence", node.confidence))
        prev_xyz = attrs.get("coordinates_3d", node.coordinates_3d)
        filtered_xyz = [
            self.momentum * float(prev_xyz[i]) + (1.0 - self.momentum) * node.coordinates_3d[i]
            for i in range(3)
        ]

        self.graph.nodes[matched_id]["confidence"] = self.momentum * prev_conf + (1.0 - self.momentum) * node.confidence
        self.graph.nodes[matched_id]["coordinates_3d"] = filtered_xyz
        return matched_id

    def _find_merge_candidate(self, node: SpatialNode) -> Optional[str]:
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") != node.label:
                continue
            xyz = attrs.get("coordinates_3d")
            if xyz is None:
                continue
            distance = sqrt(
                (float(xyz[0]) - node.coordinates_3d[0]) ** 2
                + (float(xyz[1]) - node.coordinates_3d[1]) ** 2
                + (float(xyz[2]) - node.coordinates_3d[2]) ** 2
            )
            if distance <= self.merge_distance_m:
                return str(node_id)
        return None

