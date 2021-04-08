"""
Utilities for creating typed Python dataclasses for storing hyperparameters.
"""
from typing import Mapping

from scalarstop._naming import hash_id
from scalarstop.dataclasses import asdict, dataclass, is_dataclass
from scalarstop.exceptions import (
    WrongHyperparamsKeys,
    WrongHyperparamsType,
    YouForgotTheHyperparams,
)


def enforce_dict(hyperparams):
    """Convert the input into a dictionary, whether it is a dataclass or not."""
    if hyperparams is None:
        return dict()
    if isinstance(hyperparams, dict):
        return hyperparams
    try:
        return asdict(hyperparams)
    except TypeError:
        return dict(hyperparams)


def hash_hyperparams(hyperparams):
    """Return a string hash of a given Hyperparams dataclass."""
    return hash_id(asdict(hyperparams))


def init_hyperparams(*, self, hyperparams, hyperparams_class):
    """
    Construct a hyperparams object from either a mapping or another hyperparams object.
    """
    if isinstance(hyperparams_class, type) and is_dataclass(hyperparams_class):
        if hyperparams is None:
            try:
                return hyperparams_class()
            except (TypeError, ValueError, SyntaxError) as exc:
                raise WrongHyperparamsKeys(
                    hyperparams=hyperparams, hyperparams_class=hyperparams_class
                ) from exc
        if isinstance(hyperparams, hyperparams_class):
            return hyperparams
        if isinstance(hyperparams, Mapping):
            try:
                return hyperparams_class(**hyperparams)
            except (TypeError, ValueError, SyntaxError) as exc:
                raise WrongHyperparamsKeys(
                    hyperparams=hyperparams, hyperparams_class=hyperparams_class
                ) from exc
        raise WrongHyperparamsType(hyperparams=hyperparams, obj=self)
    raise YouForgotTheHyperparams(self)


@dataclass
class HyperparamsType:
    """
    Parent class for all dataclasses containing hyperparameters.

    You will still need the ``@sp.dataclass`` decorator
    over all classes that inherit from :py:class:`HyperparamsType`.
    """


@dataclass
class NestedHyperparamsType(HyperparamsType):
    """
    Hyperparams dataclass for encapsulating the hyperparams for another model.

    When you create a dataset or model template that inherits from or combines
    other models, you should use this dataclass to store the hyperparams
    of the other model(s).
    """

    name: str
    group_name: str
    hyperparams: HyperparamsType


@dataclass
class AppendHyperparamsType(HyperparamsType):
    """
    Hyperparams for "child" datasets or models.

    Subclass this dataclass and automatically create a ``parent``
    attribute for storing the hyperparams of a parent class.
    """

    parent: NestedHyperparamsType
