"""Tools for measuring time."""
import datetime


def utcnow() -> datetime.datetime:
    """
    Returns a TIMEZONE-AWARE :py:class:`datetime.datetime` object.

    The :py:meth:`datetime.datetime.utcnow` function in
    the standard library returns a timezone-naive
    timestamp, which could easily be misinterpreted as
    being in the local (non-UTC) timezone.
    """
    return datetime.datetime.now(tz=datetime.timezone.utc)
