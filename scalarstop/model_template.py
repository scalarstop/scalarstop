"""
A class that builds models with hyperparameters.

A :py:class:`ModelTemplate` is a description of how to create a
compiled machine learning model and the hyperparameters that the
model depends on.

Subclass :py:class:`ModelTemplate` to describe the *architecture*
of your model and the hyperparameters that are used to construct
the model (and the model's optimizer).

Then, pass an instance of your :py:class:`ModelTemplate` subclass
to as subclass of :py:class:`~scalarstop.model.Model` to train an
instance of a machine learning model created from your
:py:class:`ModelTemplate`.

>>> import tensorflow as tf
>>> import scalarstop as sp
>>>
>>> class small_dense_10_way_classifier_v1(sp.ModelTemplate):
...    @sp.dataclass
...
...    class Hyperparams(sp.HyperparamsType):
...        hidden_units: int
...        optimizer: str = "adam"
...
...    def new_model(self):
...        model = tf.keras.Sequential(
...            layers=[
...                tf.keras.layers.Flatten(input_shape=(28, 28)),
...                tf.keras.layers.Dense(
...                    units=self.hyperparams.hidden_units,
...                    activation="relu",
...                ),
...                tf.keras.layers.Dense(units=10)
...           ],
...            name=self.name,
...        )
...        model.compile(
...            optimizer=self.hyperparams.optimizer,
...            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
...            metrics=["accuracy"],
...        )
...        return model

>>> model_template = small_dense_10_way_classifier_v1(hyperparams=dict(hidden_units=20))

>>> model_template.name
'small_dense_10_way_classifier_v1-zc9r3do1baeeffafanjnjmou'

"""
from typing import Any

from scalarstop._single_namespace import SingleNamespace
from scalarstop.exceptions import IsNotImplemented


class ModelTemplate(SingleNamespace):
    """Describes machine learning model architectures and hyperparameters. Used to generate new machine learning model objects that are passed into :py:class:`~scalarstop.model.Model` objects."""  # pylint: disable=line-too-long

    _model = None

    def __repr__(self) -> str:
        return f"<sp.ModelTemplate {self.name}>"

    def new_model(self) -> Any:
        """
        Create a new compiled model with the current hyperparameters.

        When you override this method, make sure to create a new model
        object every single time this function is called.
        """
        raise IsNotImplemented("ModelTemplate.new_model()")
