"""Exception types for the speceval framework."""


class SpecEvalError(Exception):
    """Base exception for all speceval errors."""


class SpecValidationError(SpecEvalError):
    """Raised when a specification fails validation."""


class SpecParseError(SpecEvalError):
    """Raised when a specification cannot be parsed."""


class ModelAdapterError(SpecEvalError):
    """Raised when a model adapter fails."""


class ModelNotFoundError(ModelAdapterError):
    """Raised when a model is not found or not accessible."""


class DatasetLoadError(SpecEvalError):
    """Raised when a dataset cannot be loaded."""


class MetricError(SpecEvalError):
    """Raised when a metric computation fails."""


class RunnerError(SpecEvalError):
    """Raised during evaluation execution."""


class StoreError(SpecEvalError):
    """Raised when the result store fails."""


class CompareError(SpecEvalError):
    """Raised when comparison operations fail."""


class ProvenanceError(SpecEvalError):
    """Raised when provenance capture fails."""
