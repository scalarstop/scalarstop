"""Common utilities for classes that manage hyperparameters."""
from typing import Any, Mapping, Optional, Type, Union

from scalarstop.hyperparams import HyperparamsType, hash_hyperparams, init_hyperparams


class SingleNamespace:
    """
    A common base class for classes with hyperparameters.

    Subclasses of this class takes hyperparameters of
    :py:class:`~scalarstop.hyperparams.HyperparamsType` and calculates
    hash-based names of the hyperparameters.
    """

    Hyperparams: Type[HyperparamsType] = HyperparamsType
    hyperparams: HyperparamsType

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
        self.hyperparams = init_hyperparams(
            class_name=self.__class__.__name__,
            hyperparams=hyperparams,
            hyperparams_class=self.Hyperparams,
        )

    @classmethod
    def calculate_name(
        cls, *, hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None
    ) -> str:
        """
        The hashed name of this object, given the hyperparameters.

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
