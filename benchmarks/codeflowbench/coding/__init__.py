"""Repository-based Coding Task regression track for CodeFlowBench.

This module stays under ``benchmarks`` and is intentionally not part of the
CodeFlow runtime package. Task commands are executable code and must come from
trusted task definitions.
"""

from .runner import CodingTask, load_task, run_task
from .trajectory import score_trace

__all__ = ["CodingTask", "load_task", "run_task", "score_trace"]
