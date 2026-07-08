from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, Iterable, List

import numpy as np


class StreamBufferSystem:
    def __init__(self, max_len: int = 32) -> None:
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_len)
        self._lock = Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(self._to_bchw(frame))

    def pop_all_context(self, latest_n: int = 5) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.empty((0, 3, 0, 0), dtype=np.float32)
            frames = list(self._buffer)[-latest_n:]
        return np.concatenate(frames, axis=0)

    @staticmethod
    def _to_bchw(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim != 4:
            raise ValueError("Unsupported frame shape. Expected HWC, CHW, NHWC, or NCHW.")

        return arr

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def snapshot(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._buffer)

    def extend(self, frames: Iterable[np.ndarray]) -> None:
        with self._lock:
            for frame in frames:
                self._buffer.append(self._to_bchw(frame))

