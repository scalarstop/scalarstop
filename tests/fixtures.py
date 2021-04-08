"""Fixtures for ScalarStop tests"""
import unittest

import tensorflow as tf

import scalarstop as sp

requires_sqlite_json = unittest.skipIf(
    not sp.train_store.sqlite_json_enabled(),
    "The SQLite3 JSON1 extension is not enabled in this Python installation.",
)


class MyDataBlob(sp.DataBlob):
    """An example DataBlob for training."""

    @sp.dataclass
    class Hyperparams:
        """Hyperparams."""

        rows: int
        cols: int

    def _tfdata(self):
        """Generate example data."""
        x = tf.random.uniform(
            shape=(self.hyperparams.rows, self.hyperparams.cols), dtype=tf.float32
        )
        y = tf.random.uniform(shape=(self.hyperparams.rows,), dtype=tf.float32)
        return tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(x),
                tf.data.Dataset.from_tensor_slices(y),
            )
        )

    def set_training(self):
        return self._tfdata()

    def set_validation(self):
        return self._tfdata()

    def set_test(self):
        return self._tfdata()


class MyModelTemplate(sp.ModelTemplate):
    """Example model template."""

    @sp.dataclass
    class Hyperparams:
        """Hyperparams."""

        layer_1_units: int
        optimizer: str = "adam"
        loss: str = "binary_crossentropy"

    def new_model(self):
        """Set a model."""
        model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(
                    units=self.hyperparams.layer_1_units,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
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
            optimizer=self.hyperparams.optimizer,
            loss=self.hyperparams.loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model
