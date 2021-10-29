import tempfile
import unittest

import scalarstop as sp
import tensorflow as tf
import tensorflow.python as tfpy

from tests.fixtures import MyDataBlob


try:
    import keras
except ImportError:
    _REVIVED_LAYER_CLASSES = (
        tfpy.keras.saving.saved_model.load.RevivedLayer,
    )
else:
    _REVIVED_LAYER_CLASSES = (
        keras.saving.saved_model.load.RevivedLayer,
        tfpy.keras.saving.saved_model.load.RevivedLayer,
    )



class AddByConstant(tf.keras.layers.Layer):
    def __init__(self, *, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def get_config(self):
        return dict(
            **super().get_config(),
            constant=self.constant,
        )

    def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
        return inputs + self.constant


class MultiplyByConstant(tf.keras.layers.Layer):
    def __init__(self, *, constant: float, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def get_config(self):
        return dict(
            **super().get_config(),
            constant=self.constant,
        )

    def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
        return inputs * self.constant


def new_keras_model(
    *,
    multiply_constant: float,
    add_constant: float,
    layer_1_units: int,
    optimizer: str,
    loss: str,
):
    model = tf.keras.Sequential(
        layers=[
            MultiplyByConstant(
                constant=multiply_constant,
            ),
            tf.keras.layers.Dense(
                units=layer_1_units,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
            AddByConstant(
                constant=add_constant,
            ),
            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class ModelTemplateCustomLayers(sp.ModelTemplate):
    """Example model template."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams."""

        multiply_constant: float = 2.0
        add_constant: float = 4.0
        layer_1_units: int = 3
        optimizer: str = "adam"
        loss: str = "binary_crossentropy"

    def new_model(self):
        """Set a model."""
        return new_keras_model(
            multiply_constant=self.hyperparams.multiply_constant,
            add_constant=self.hyperparams.add_constant,
            layer_1_units=self.hyperparams.layer_1_units,
            optimizer=self.hyperparams.optimizer,
            loss=self.hyperparams.loss,
        )


class KerasModelTemplateCustomLayersCustomObjects(sp.KerasModelTemplate):
    """Example model template."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams."""

        multiply_constant: float = 2.0
        add_constant: float = 4.0
        layer_1_units: int = 3
        optimizer: str = "adam"
        loss: str = "binary_crossentropy"

    @property
    def custom_objects(self):
        return dict(
            AddByConstant=AddByConstant,
            MultiplyByConstant=MultiplyByConstant,
        )

    def new_model(self):
        """Set a model."""
        return new_keras_model(
            multiply_constant=self.hyperparams.multiply_constant,
            add_constant=self.hyperparams.add_constant,
            layer_1_units=self.hyperparams.layer_1_units,
            optimizer=self.hyperparams.optimizer,
            loss=self.hyperparams.loss,
        )


class TestModelCustomLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datablob = MyDataBlob(hyperparams=dict(rows=10, cols=5)).batch(2)

    def setUp(self):
        self.temp_dir_context = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.models_directory = self.temp_dir_context.name

    def tearDown(self):
        self.temp_dir_context.cleanup()

    def test_save_load_without_custom_objects(self):
        model_template = ModelTemplateCustomLayers()
        model = sp.KerasModel(
            datablob=self.datablob,
            model_template=model_template,
        )
        model.save(self.models_directory)

        loaded_model = sp.KerasModel.from_filesystem(
            datablob=self.datablob,
            model_template=model_template,
            models_directory=self.models_directory,
        )
        multiply_by_constant_layer = loaded_model.model.layers[0]
        add_by_constant_layer =  loaded_model.model.layers[2]
        self.assertIsInstance(multiply_by_constant_layer, _REVIVED_LAYER_CLASSES)
        self.assertIsInstance(add_by_constant_layer, _REVIVED_LAYER_CLASSES)