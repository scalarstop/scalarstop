"""Utilities for probing information about the current machine's CPUs."""
import os
from typing import Optional


def num_usable_virtual_cpu_cores() -> Optional[int]:
    """
    Returns the number of virtual CPU cores available to the
    current process.

    This function uses :py:func:`os.sched_getaffinity`
    to check when only a subset of all virtual CPU cores are
    available to this process. Currently, this only works
    on Linux.

    On other platforms, this function will return the total number
    of virtual CPU cores on this machine.

    If this function cannot find the total number of cores,
    it will return ``None``.

    This function returns ``None`` if it was not able to
    find the number of CPU cores on this machine.

    Returns:
        An integer number of virtual CPU cores or ``None``.
    """
    try:
        # Here we check the number of virtual CPU cores that
        # can be used by the current process, which may be less than the
        # total number of virtual CPU cores on the machine.
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
