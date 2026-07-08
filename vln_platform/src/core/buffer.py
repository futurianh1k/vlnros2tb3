from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Deque, Optional

import numpy as np


@dataclass(slots=True)
class BufferEntry:
    tensor: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamBufferSystem:
    def __init__(self, max_len: int = 128) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self._buffer: Deque[BufferEntry] = deque(maxlen=max_len)
        self._lock = Lock()

    @staticmethod
    def _normalize_to_batch_chw_tensor(image: Any) -> np.ndarray:
        tensor = np.asarray(image)

        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, np.newaxis, :, :]
        elif tensor.ndim == 3:
            if tensor.shape[0] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
                tensor = tensor[np.newaxis, :, :, :]
            elif tensor.shape[-1] in {1, 3, 4}:
                tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :, :, :]
            else:
                tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :, :, :]
        elif tensor.ndim == 4:
            if tensor.shape[1] in {1, 3, 4}:
                pass
            elif tensor.shape[-1] in {1, 3, 4}:
                tensor = np.transpose(tensor, (0, 3, 1, 2))
            else:
                raise ValueError("Unsupported 4D tensor layout")
        else:
            raise ValueError("Unsupported tensor rank for image input")

        return np.ascontiguousarray(tensor)

    def push(self, image: Any, metadata: Optional[dict[str, Any]] = None) -> BufferEntry:
        entry = BufferEntry(
            tensor=self._normalize_to_batch_chw_tensor(image),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._buffer.append(entry)
        return entry

    def pop_all_context(self) -> list[BufferEntry]:
        with self._lock:
            entries = list(self._buffer)
            self._buffer.clear()
        return entries

    def latest_batch(self, batch_size: int = 5) -> Optional[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            selected = list(self._buffer)[-batch_size:]
        if not selected:
            return None
        tensors = [entry.tensor for entry in selected]
        reference_shape = tensors[0].shape[1:]
        for tensor in tensors[1:]:
            if tensor.shape[1:] != reference_shape:
                raise ValueError("All buffered tensors must share the same spatial shape")
        return np.concatenate(tensors, axis=0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


__all__ = ["BufferEntry", "StreamBufferSystem"]
