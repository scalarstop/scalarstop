"""
Utilities for creating typed Python dataclasses for storing hyperparameters.
"""
from typing import TYPE_CHECKING, Any, Dict, Mapping

from scalarstop._attr_dict import AttrDict
from scalarstop._naming import hash_id
from scalarstop.dataclasses import asdict, is_dataclass
from scalarstop.exceptions import WrongHyperparamsType, YouForgotTheHyperparams

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from scalarstop.dataclasses import dataclass


def enforce_dict(hyperparams: Any) -> Dict[Any, Any]:
    """Convert the input into a dictionary, whether it is a dataclass or not."""
    if hyperparams is None:
        return {}
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
            return hyperparams_class()
        if isinstance(hyperparams, hyperparams_class):
            return hyperparams
        if isinstance(hyperparams, Mapping):
            return hyperparams_class(**hyperparams)
        raise WrongHyperparamsType(hyperparams=hyperparams, class_name=class_name)
    raise YouForgotTheHyperparams(class_name=class_name)


def _flatten_hyperparams(hp_dict):
    parent_hp_dict = hp_dict.pop("parent", None)
    if parent_hp_dict:
        return {
            **_flatten_hyperparams(parent_hp_dict["hyperparams"]),
            **hp_dict,
        }
    return hp_dict


def flatten_hyperparams(hyperparams: Any) -> Dict[str, Any]:
    """
    Recursively flatten the hyperparams embedded
    in a :py:class:`NestedHyperparamsType` instance.
    """
    return AttrDict(**_flatten_hyperparams(enforce_dict(hyperparams)))


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
