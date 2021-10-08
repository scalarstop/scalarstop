"""Issue warnings."""
import warnings as _python_warnings


class ScalarStopWarning(UserWarning):
    """Parent class for all ScalarStop warnings."""


class ScalarStopDeprecationWarning(ScalarStopWarning):
    """Raised when the user uses a deprecated feature."""


def warn(message: str) -> None:
    """Issue a ScalarStop warning."""
    _python_warnings.warn(message=message, category=ScalarStopWarning, stacklevel=2)


def warn_deprecated(message: str) -> None:
    """Issue a ScalarStop deprecation warning."""
    _python_warnings.warn(
        message=message, category=ScalarStopDeprecationWarning, stacklevel=2
    )
