"""Fixtures for ScalarStop tests"""
import os
import unittest
from typing import Any, Mapping, Optional, Union

import tensorflow as tf

import scalarstop as sp

requires_external_database = unittest.skipUnless(
    os.environ.get("TRAIN_STORE_CONNECTION_STRING", False),
    "External database connection string was not supplied.",
)

requires_sqlite_json = unittest.skipIf(
    not sp.train_store._sqlite_json_enabled(),
    "The SQLite3 JSON1 extension is not enabled in this Python installation.",
)


class MyDataBlob(sp.DataBlob):
    """An example DataBlob for training."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
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


class MyDataBlobRepeating(MyDataBlob):
    """An infinitely-repeating example DataBlob."""

    def _tfdata(self):
        return super()._tfdata().repeat()


class MyDataBlob2(MyDataBlob):
    """Another DataBlob to test different group names."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
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


class MyShardableDataBlob(MyDataBlob):
    """
    A :py:class:`~scalarstop.datablob.DataBlob` instance that
    handles sharding internally.
    """

    def __init__(
        self,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], sp.HyperparamsType]] = None,
        num_shards: Optional[int] = None,
        shard_index: Optional[int] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(hyperparams=hyperparams, **kwargs)
        self._num_shards = num_shards
        self._shard_index = shard_index

    def _tfdata(self):
        if self._num_shards is not None and self._shard_index is not None:
            return (
                super()
                ._tfdata()
                .shard(num_shards=self._num_shards, index=self._shard_index)
            )
        return super()._tfdata()


class MyModelTemplate(sp.ModelTemplate):
    """Example model template."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
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


class MyModelTemplate2(MyModelTemplate):
    """Another ModelTemplate to test different group names."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
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


class MyShardableDistributedDataBlob(sp.DistributedDataBlob):
    """
    An example of a custom :py:class:`~scalarstop.datablob.DistributedDataBlob`
    subclass, wrapping :py:class:`MyShardableDataBlob`.
    """

    def __init__(
        self,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], sp.HyperparamsType]] = None,
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.get_strategy] = None,
    ):
        name = MyShardableDataBlob.calculate_name(hyperparams=hyperparams)
        group_name = MyShardableDataBlob.__name__
        hyperparams_class = MyShardableDataBlob.Hyperparams
        super().__init__(
            name=name,
            group_name=group_name,
            hyperparams=hyperparams,
            hyperparams_class=hyperparams_class,
            repeat=repeat,
            per_replica_batch_size=per_replica_batch_size,
            tf_distribute_strategy=tf_distribute_strategy,
        )

    def new_sharded_datablob(
        self, ctx: tf.distribute.InputContext  # pylint: disable=unused-argument
    ) -> sp.DataBlob:
        return MyShardableDataBlob(
            hyperparams=self._hyperparams,
            num_shards=ctx.num_input_pipelines,
            shard_index=ctx.input_pipeline_id,
        )
