#!/usr/bin/env python3
"""
ModelWorker - Isolated process for PyTorch model loading and inference.

ONE UNIVERSE: Legacy checkpoint-based worker (16/88, session/vol/trend) removed.
Inference uses ctx bundle loaded via model_loader_worker and in-process forward
(seq_x, snap_x, ctx_cat, ctx_cont). This module now hard-fails if used.
"""

import logging
import os
import sys
from typing import Any

from gx1.inference.protocol import InferenceRequest, InferenceResponse, WorkerConfig

log = logging.getLogger(__name__)


def worker_main(
    config: WorkerConfig,
    request_queue: Any,  # multiprocessing.Queue
    response_queue: Any,  # multiprocessing.Queue
) -> None:
    """
    Main entry for ModelWorker process.
    ONE UNIVERSE: Only v10_ctx bundle inference is supported. This path is removed.
    """
    log.info("[MODEL_WORKER] Starting worker process (PID=%d)", os.getpid())
    raise RuntimeError(
        "ONE_UNIVERSE: Legacy ModelWorker (checkpoint + session/vol/trend) removed. "
        "Use model_loader_worker to load v10_ctx bundle and run inference in-process with ctx_cat/ctx_cont (6/6)."
    )


if __name__ == "__main__":
    # Standalone entry point for testing (not used in production)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.error("[MODEL_WORKER] This module should not be run directly. Use gx1.inference.client instead.")
    sys.exit(1)

