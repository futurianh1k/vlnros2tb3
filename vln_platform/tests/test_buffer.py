from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.buffer import StreamBufferSystem


def test_buffer_returns_bchw_context_with_thread_safe_pushes():
    buffer = StreamBufferSystem(max_len=8)

    def _push_frames(value: int) -> None:
        for _ in range(5):
            frame = np.full((4, 4, 3), fill_value=value, dtype=np.uint8)
            buffer.push(frame)

    threads = [Thread(target=_push_frames, args=(idx,)) for idx in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    context = buffer.pop_all_context(latest_n=5)
    assert context.ndim == 4
    assert context.shape[1] == 3
    assert context.shape[0] <= 5

