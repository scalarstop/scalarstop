"""
A class that builds models with hyperparameters.

A :py:class:`ModelTemplate` is a description of how to create a
compiled machine learning model and the hyperparameters that the
model depends on.
"""
from typing import Any, Mapping, Optional, Union

from scalarstop.exceptions import IsNotImplemented
from scalarstop.hyperparams import HyperparamsType, hash_hyperparams, init_hyperparams


class ModelTemplate:
    """Describes machine learning model architectures and hyperparameters. Used to generate new machine learning model objects that are passed into :py:class:`~scalarstop.Model` objects."""  # pylint: disable=line-too-long

    Hyperparams: type
    hyperparams: HyperparamsType

    _name: Optional[str] = None
    _group_name: Optional[str] = None
    _model = None

    def __init__(
        self, *, hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None
    ):
        self.hyperparams = init_hyperparams(
            self=self,
            hyperparams=hyperparams,
            hyperparams_class=self.Hyperparams,
        )

    def __repr__(self) -> str:
        return f"<sp.ModelTemplate {self.name}>"

    @property
    def name(self) -> str:
        """
        The name of the specific model with values as hyperparameters.

        If you intend on overriding this method, make sure that
        instances of the same class with different hyperparameters
        will have different names.

        However, if you use additional parameters to your
        :py:class:`ModelTemplate` 's ``__init__()`` method that are
        *not* hyperparameters--such as paths on your filesystem--then
        you should be sure that changes in those values do *not*
        change your :py:class:`ModelTemplate` name.
        """
        if self._name is None:
            self._name = "-".join((self.group_name, hash_hyperparams(self.hyperparams)))
        return self._name

    @property
    def group_name(self) -> str:
        """
        The group name of this model template.

        Conceptually, the group name is the name for all compiled machine
        learning models that share the same code but have different
        hyperparameters.
        """
        if self._group_name is None:
            self._group_name = self.__class__.__name__
        return self._group_name

    def new_model(self) -> Any:
        """
        Create a new compiled model with the current hyperparameters.

        When you override this method, make sure to create a new model
        object every single time this function is called.
        """
        raise IsNotImplemented("ModelTemplate.new_model()")
