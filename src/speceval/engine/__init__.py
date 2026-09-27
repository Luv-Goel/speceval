"""Engine — orchestration of evaluation runs."""

from speceval.engine.runner import EvaluationRunner, get_runner
from speceval.engine.task import EvalResult, EvalTask

__all__ = [
    "EvalResult",
    "EvalTask",
    "EvaluationRunner",
    "get_runner",
]
