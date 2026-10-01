"""Recoverable model-output failures, distinct from service and data errors."""
class GenerationError(ValueError):
    pass
