from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.core.buffer import StreamBufferSystem


def test_stream_buffer_is_thread_safe_and_normalizes_layout() -> None:
    buffer = StreamBufferSystem(max_len=4)

    hwc_frame = np.ones((8, 6, 3), dtype=np.uint8)
    chw_frame = np.ones((3, 8, 6), dtype=np.uint8)

    def push_frame(frame: np.ndarray) -> None:
        buffer.push(frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(3):
            executor.submit(push_frame, hwc_frame)
            executor.submit(push_frame, chw_frame)

    latest = buffer.latest_batch(batch_size=4)
    assert latest is not None
    assert latest.shape == (4, 3, 8, 6)

    entries = buffer.pop_all_context()
    assert len(entries) == 4
    assert all(entry.tensor.shape == (1, 3, 8, 6) for entry in entries)
