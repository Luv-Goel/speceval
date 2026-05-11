"""Exception types for the speceval framework."""


class SpecEvalError(Exception):
    """Base exception for all speceval errors.

    Adds a prefixed ``__str__`` so that log lines produced by any
    sub-class are immediately greppable::

        [SpecValidationError] field 'metric' is required

    Sub-classes that need additional structured fields (e.g. a spec path
    or model name) should call ``super().__init__(message)`` and store
    the extra data as instance attributes rather than embedding them in
    the message string.
    """

    def __str__(self) -> str:  # pragma: no cover
        tag = type(self).__name__
        msg = super().__str__()
        # Avoid double-tagging if the message was already tagged (e.g.
        # when an exception is re-raised with str(original_exc) as the
        # new message).
        if msg.startswith(f"[{tag}]"):
            return msg
        return f"[{tag}] {msg}" if msg else f"[{tag}]"

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({super().__str__()!r})"


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
