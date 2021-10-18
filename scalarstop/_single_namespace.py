"""Common utilities for classes that manage hyperparameters."""
from typing import Any, Dict, Mapping, Optional, Type, Union

from scalarstop.hyperparams import (
    HyperparamsType,
    flatten_hyperparams,
    hash_hyperparams,
    init_hyperparams,
)


class SingleNamespace:
    """
    A common base class for classes with hyperparameters.

    Subclasses of this class takes hyperparameters of
    :py:class:`~scalarstop.hyperparams.HyperparamsType` and calculates
    hash-based names of the hyperparameters.
    """

    Hyperparams: Type[HyperparamsType] = HyperparamsType

    _group_name: str = ""
    _name: str = ""

    def __init__(
        self,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Args:
            hyperparams: The hyperparameters to initialize this class with.
        """
        self._hyperparams = init_hyperparams(
            class_name=self.__class__.__name__,
            hyperparams=hyperparams,
            hyperparams_class=self.Hyperparams,
        )
        self._hyperparams_flat = flatten_hyperparams(self._hyperparams)

    @classmethod
    def calculate_name(
        cls, *, hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None
    ) -> str:
        """
        Calculate the hashed name of this object, given the hyperparameters.

        This classmethod can be used to calculate what an object would
        be without actually having to call ``__init__()``.
        """
        hp = init_hyperparams(
            class_name=cls.__name__,
            hyperparams=hyperparams,
            hyperparams_class=cls.Hyperparams,
        )
        return "-".join((cls.__name__, hash_hyperparams(hp)))

    @property
    def group_name(self) -> str:
        """The "group" name is this object's Python class name."""
        if not self._group_name:
            self._group_name = self.__class__.__name__
        return self._group_name

    @property
    def name(self) -> str:
        """The group (class) name and a calculated hash of the hyperparameters."""
        if not self._name:
            self._name = self.calculate_name(hyperparams=self.hyperparams)
        return self._name

    @property
    def hyperparams(self) -> HyperparamsType:
        """Returns a :py:class:`HyperparamsType` instance containing hyperparameters."""
        return self._hyperparams

    @property
    def hyperparams_flat(self) -> Dict[str, Any]:
        """
        Returns a Python dictionary of "flattened" hyperparameters.

        :py:class:`AppendDataBlob` objects modify a
        "parent" :py:class:`DataBlob`, nesting the parent's
        `Hyperparams` within the :py:class:`AppendDataBlob` 's
        own `Hyperparams`.

        This makes it hard to look up a given hyperparams
        key. A value at ``parent_datablob.hyperparams.a`` is
        stored at ``child_datablob.hyperparams.parent.hyperparams.a``.

        This ``hyperparams_flat`` property provides all
        nested hyperparams keys as a flat Python dictionary.
        If a child :py:class:`AppendDataBlob` has a hyperparameter
        key that that conflicts with the parent, the child's value
        will overwrite the parent's value.
        """
        return self._hyperparams_flat
