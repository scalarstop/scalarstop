"""Utilities for probing information about the current machine's CPUs."""
import os
from typing import Optional

from log_with_context import Logger

_LOGGER = Logger(__name__)


def num_usable_virtual_cpu_cores(default_to_all: bool = True) -> Optional[int]:
    """
    Returns the number of virtual CPU cores available to the
    current process.

    This function returns ``None`` if it was not able to
    find the number of CPU cores on this machine.

    Args:
        default_to_all: Defaults to ``True``, which makes
            this function return the *total* number of
            virtual CPU cores on this machine if it cannot
            find the number of virtual CPU cores
            *specifically available* to this process. If we
            cannot find the total number of virtual cores,
            then this function still returns ``None``.
            If you set ``default_to_all`` to ``False``,
            then this function raises an exception.

    Returns:
        An integer number of virtual CPU cores or ``None``.
    """
    try:
        # Here we check the number of virtual CPU cores that
        # can be used by the current process, which may be less than the
        # total number of virtual CPU cores on the machine.
        return len(os.sched_getaffinity(0))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # pylint: disable=broad-except
        if not default_to_all:
            raise
        _LOGGER.exception(
            "Failed to find the number of available CPUs using "
            "`os.sched_getaffinity(0)`."
        )
        return os.cpu_count()
