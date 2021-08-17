"""Secret logging utilities that will help out."""
import time
from typing import Callable, Optional

from log_with_context import Logger

_LOGGER = Logger(__name__)


class Timeblock:
    """
    Context manager that prints out how long the containing code took to execute.

    Usage:
        import time
        with Timeblock(name="helloworld"):
            time.sleep(5)
            print("Hello world")

    If you want to send a custom logging function, do as follows:
        import logging
        import time
        with Timeblock(name="helloworld", print_function=logging.debug):
            time.sleep(5)
            print("Hello world")
    """

    def __init__(
        self,
        *,
        name: str,
        print_function: Optional[Callable] = None,
        print_start: bool = True,
    ):
        """
        Args:
            name: The name of this timer. We print it at the beginning
                of every logging message.

            print_function: A Python callable that can accept a single
                string logging argument.

            print_start: Set this ``False` to suppress the logging message
                at the start of the timing process.
        """
        self.name = name
        self.print_function = print_function or _LOGGER.info
        self.print_start = print_start
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.print_start:
            self.print_function(
                f"{self.name}: beginning timer now at unix time {time.time()}."
            )
        self.start_time = time.monotonic()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.monotonic()
        elapsed = self.end_time - self.start_time
        if exc_type:
            exc_name = exc_type.__name__
            self.print_function(
                f"{self.name}: {elapsed} seconds until INTERRUPTED by exception {exc_name}."
            )
        else:
            self.print_function(f"{self.name}: {elapsed} seconds elapsed.")
