"""
A wrapper for cloudpickle to (de)serialize Python objects.
"""

import cloudpickle

_PICKLE_PROTOCOL_VERSION = 4

_PICKLE_FIX_IMPORTS = True

_PICKLE_ENCODING = "ASCII"

_PICKLE_ERRORS = "strict"


def load(file):
    """Load a Python object from a file handle."""
    return cloudpickle.load(
        file=file,
        fix_imports=_PICKLE_FIX_IMPORTS,
        encoding=_PICKLE_ENCODING,
        errors=_PICKLE_ERRORS,
    )


def loads(data):
    """Load a Python object from a string."""
    return cloudpickle.loads(
        data=data,
        fix_imports=_PICKLE_FIX_IMPORTS,
        encoding=_PICKLE_ENCODING,
        errors=_PICKLE_ERRORS,
    )


def dump(*, obj, file):
    """Dump a Python object to a file handle."""
    return cloudpickle.dump(
        obj=obj,
        file=file,
        protocol=_PICKLE_PROTOCOL_VERSION,
    )


def dumps(obj):
    """Dump a Python object to a string."""
    return cloudpickle.dumps(
        obj=obj,
        protocol=_PICKLE_PROTOCOL_VERSION,
    )
