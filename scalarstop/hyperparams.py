"""
Utilities for creating typed Python dataclasses for storing hyperparameters.
"""
from typing import TYPE_CHECKING, Any, Dict, Mapping

from scalarstop._naming import hash_id
from scalarstop.dataclasses import asdict, is_dataclass
from scalarstop.exceptions import (
    WrongHyperparamsKeys,
    WrongHyperparamsType,
    YouForgotTheHyperparams,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from scalarstop.dataclasses import dataclass


def enforce_dict(hyperparams: Any) -> Dict[Any, Any]:
    """Convert the input into a dictionary, whether it is a dataclass or not."""
    if hyperparams is None:
        return dict()
    if isinstance(hyperparams, dict):
        return hyperparams
    try:
        return asdict(hyperparams)
    except TypeError:
        return dict(hyperparams)


def hash_hyperparams(hyperparams: Any) -> str:
    """Return a string hash of a given Hyperparams dataclass."""
    return hash_id(asdict(hyperparams))


def init_hyperparams(*, class_name: str, hyperparams, hyperparams_class) -> Any:
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
        raise WrongHyperparamsType(hyperparams=hyperparams, class_name=class_name)
    raise YouForgotTheHyperparams(class_name=class_name)


@dataclass  # pylint: disable=used-before-assignment
class HyperparamsType:
    """
    Parent class for all dataclasses containing hyperparameters.

    You will still need the ``@sp.dataclass`` decorator
    over all classes that inherit from :py:class:`HyperparamsType`.
    """


@dataclass  # pylint: disable=used-before-assignment
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


@dataclass  # pylint: disable=used-before-assignment
class AppendHyperparamsType(HyperparamsType):
    """
    Hyperparams for "child" datasets or models.

    Subclass this dataclass and automatically create a ``parent``
    attribute for storing the hyperparams of a parent class.
    """

    parent: NestedHyperparamsType
