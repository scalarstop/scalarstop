"""
A wrapper for cloudpickle to (de)serialize Python objects.
"""

from typing import Any, BinaryIO, TextIO, Union, cast

import cloudpickle

_PICKLE_PROTOCOL_VERSION = 4

_PICKLE_FIX_IMPORTS = True

_PICKLE_ENCODING = "ASCII"

_PICKLE_ERRORS = "strict"


def load(file: Union[BinaryIO, TextIO]) -> Any:
    """Load a Python object from a file handle."""
    return cloudpickle.load(
        file=file,
        fix_imports=_PICKLE_FIX_IMPORTS,
        encoding=_PICKLE_ENCODING,
        errors=_PICKLE_ERRORS,
    )


def loads(data: str) -> Any:
    """Load a Python object from a string."""
    return cloudpickle.loads(
        data=data,
        fix_imports=_PICKLE_FIX_IMPORTS,
        encoding=_PICKLE_ENCODING,
        errors=_PICKLE_ERRORS,
    )


def dump(*, obj: Any, file: Union[BinaryIO, TextIO]) -> None:
    """Dump a Python object to a file handle."""
    cloudpickle.dump(
        obj=obj,
        file=file,
        protocol=_PICKLE_PROTOCOL_VERSION,
    )


def dumps(obj: Any) -> str:
    """Dump a Python object to a string."""
    dumped = cloudpickle.dumps(
        obj=obj,
        protocol=_PICKLE_PROTOCOL_VERSION,
    )
    return cast(str, dumped)
