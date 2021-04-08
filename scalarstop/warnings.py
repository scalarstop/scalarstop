"""Issue warnings."""
import warnings as _python_warnings


class ScalarStopWarning(UserWarning):
    """Parent class for all ScalarStop warnings."""


def warn(message: str):
    """Issue a ScalarStop warning."""
    _python_warnings.warn(message=message, category=ScalarStopWarning)
