from __future__ import annotations

import numpy as np


def run_fp8_forward_pass(frame_tensor: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
    if frame_tensor.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    frame_score = float(np.mean(frame_tensor))
    text_score = float(np.mean(text_embedding)) if text_embedding.size else 0.0
    return np.array([frame_score * 0.01, text_score * 0.01], dtype=np.float32)

