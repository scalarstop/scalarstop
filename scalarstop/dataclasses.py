"""
A forked version of the Python :py:mod:`dataclasses` module.

Create a dataclass with the decorator ``@sp.dataclass`` as follows::

    import scalarstop as sp

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        val1: int
        val2: str


The Python :py:mod:`dataclasses` module uses singletons to compare
certain values in internal code. This makes it difficult to
:py:mod:`cloudpickle` a dataclass and then unpickle it.

This version of the :py:mod:`dataclasses` module is better-designed
to be shared across different Python processes.
"""

import copy
import dataclasses as _python_dataclasses
from typing import Any, Dict, Tuple, cast

_FIELD = "_FIELD"
_FIELD_CLASSVAR = "_FIELD_CLASSVAR"
_FIELD_INITVAR = "_FIELD_INITVAR"

# The name of an attribute on the class where we store the Field
# objects.  Also used to check if a class is a Data Class.
_FIELDS = "__dataclass_fields__"

dataclass = _python_dataclasses.dataclass(frozen=True)


def is_dataclass(obj: Any) -> bool:
    """Returns True if the input is a Python dataclass."""
    return _python_dataclasses.is_dataclass(obj)


def fields(class_or_instance: Any) -> Tuple[Any, ...]:
    """
    Forked verson of :py:func:`dataclasses.fields`.

    Return a tuple describing the fields of this dataclass.
    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """

    # Might it be worth caching this, per class?
    try:
        the_fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError(  # pylint: disable=raise-missing-from
            "must be called with a dataclass type or instance"
        )

    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(f for f in the_fields.values() if repr(f._field_type) == _FIELD)


def _is_dataclass_instance(obj: Any) -> bool:
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def asdict(obj: Any, *, dict_factory: type = dict) -> Dict[str, Any]:
    """
    Forked verson of :py:func:`dataclasses.asdict`.

    Return the fields of a :py:func:`~dataclasses.dataclass`
    instance as a new tuple of field values.

    Example usage::

        @dataclass
        class C:
            x: int
            y: int

      c = C(1, 2)

      assert asdict(c) == {'x': 1, 'y': 2}

    If given, ``dict_factory`` will be used instead of built-in
    :py:class:`dict`.

    The function applies recursively to field values that are
    :py:func:`~dataclasses.dataclass` instances. This will also
    look into built-in containers:
    :py:class:`tuple`, :py:class:`list`, and :py:class:`dict`.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return cast(Dict[str, Any], _asdict_inner(obj, dict_factory))


def _asdict_inner(obj, dict_factory):
    if _is_dataclass_instance(obj):  # pylint: disable=no-else-return
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
            for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


def astuple(obj, *, tuple_factory=tuple) -> Tuple[Any, ...]:
    """
    Forked verson of :py:func:`dataclasses.astuple`.

    Return the fields of a :py:func:`~dataclasses.dataclass`
    instance as a new tuple of field values.

    Example usage::

        @dataclass
        class C:
            x: int
            y: int

        c = C(1, 2)
        assert astuple(c) == (1, 2)

    If given, ``tuple_factory`` will be used instead of built-in
    :py:class:`tuple`.

    The function applies recursively to field values that are
    :py:func:`~dataclasses.dataclass` instances. This will also
    look into built-in containers:
    :py:class:`tuple`, :py:class:`list`, and :py:class:`dict`.
    """

    if not _is_dataclass_instance(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return _astuple_inner(obj, tuple_factory)


def _astuple_inner(obj, tuple_factory):
    if _is_dataclass_instance(obj):  # pylint: disable=no-else-return
        result = []
        for f in fields(obj):
            value = _astuple_inner(getattr(obj, f.name), tuple_factory)
            result.append(value)
        return tuple_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).
        return type(obj)(*[_astuple_inner(v, tuple_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_astuple_inner(v, tuple_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_astuple_inner(k, tuple_factory), _astuple_inner(v, tuple_factory))
            for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)
