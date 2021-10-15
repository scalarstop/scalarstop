"""Testing utilities that we'll use throughout the project."""
import os
import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

import scalarstop as sp
from scalarstop._constants import _DEFAULT_SAVE_LOAD_VERSION

_TESTCASE = unittest.TestCase()

assert_equal = _TESTCASE.assertEqual


def tfdata_get_first(tfdata: tf.data.Dataset) -> tf.Tensor:
    """Get the first element in a tf.data.Dataset."""
    for first in tfdata.take(1):
        return first


def tfdata_get_first_shape_len(tfdata: tf.data.Dataset) -> int:
    """Get the length of the TensorShape of the first element in a tf.data.Dataset."""
    return len(tfdata_get_first(tfdata).shape)


def assert_directory(dirname: str, expected_filenames: List[str]):
    """Assert the contents of an arbitrary directory on disk."""
    actual_filenames = os.listdir(dirname)
    _TESTCASE.assertEqual(sorted(expected_filenames), sorted(actual_filenames))


def assert_dataframes_are_equal(df1, df2):
    """Assert that two :py:class:`pd.DataFrame` s are equal."""
    equality = df1 == df2
    if isinstance(equality, bool):
        assert equality
    elif isinstance(equality, pd.DataFrame):
        assert equality.all(axis=None)
    else:
        raise TypeError(f"Unsupported equality type {equality} {type(equality)}")


def assert_hyperparams_are_equal(h1: sp.HyperparamsType, h2: sp.HyperparamsType):
    """Assert that two Hyperparams dataclasses are equal."""
    assert_equal(
        sp.dataclasses.asdict(h1),
        sp.dataclasses.asdict(h2),
    )


def assert_hyperparams_flat_are_equal(h1: Dict[str, Any], h2: Dict[str, Any]):
    """Assert the outputs of hyperparams_flat() are equal."""
    assert_equal(h1, h2)


def tfdata_as_list(tfdata: tf.data.Dataset) -> List[np.ndarray]:
    """Return a tf.data pipeline as a list of NumPy arrays."""
    return list(tfdata.as_numpy_iterator())


def assert_tfdatas_are_equal(d1: tf.data.Dataset, d2: tf.data.Dataset):
    """
    Assert that two n-dimensional :py:class:`tf.data.Dataset` are sequences of "equal elements."
    """
    l1 = tfdata_as_list(d1)
    l2 = tfdata_as_list(d2)
    assert_equal(len(l1), len(l2))
    for i1, i2 in zip(l1, l2):
        assert np.array_equal(i1, i2)


def assert_datablob_reprs_are_equal(blob1: sp.DataBlob, blob2: sp.DataBlob):
    """Assert that DataBlob __repr__() methods return the same string."""
    assert_equal(repr(blob1), repr(blob2))


def assert_datablob_names_and_hyperparams_are_equal(
    blob1: sp.DataBlob, blob2: sp.DataBlob
):
    """
    Assert that DataBlob names and hyperparams are equal.

    This assertion does not check DataFrames or Datasets.
    """
    assert_equal(repr(blob1), repr(blob2))
    assert_equal(blob1.name, blob2.name)
    assert_equal(blob1.group_name, blob2.group_name)
    assert_hyperparams_are_equal(blob1.hyperparams, blob2.hyperparams)
    assert_hyperparams_flat_are_equal(blob1.hyperparams_flat, blob2.hyperparams_flat)


def assert_datablob_metadata_from_filesystem(
    blob: sp.DataBlob,
    *,
    datablobs_directory: str,
    save_load_version: int = _DEFAULT_SAVE_LOAD_VERSION,
    num_shards: int = 1,
):
    """
    Assert that a DataBlob's DataBlobMetadata has specific values.
    """
    metadata = blob.metadata_from_filesystem(
        hyperparams=blob.hyperparams,
        datablobs_directory=datablobs_directory,
    )
    metadata_dict = metadata.to_dict(hyperparams_as_dict=False)
    assert_equal(
        sorted(list(metadata_dict.keys())),
        ["group_name", "hyperparams", "name", "num_shards", "save_load_version"],
    )
    assert_equal(
        blob.name,
        metadata.name,
    )
    assert_equal(
        blob.name,
        metadata_dict["name"],
    )
    assert_equal(
        blob.group_name,
        metadata.group_name,
    )
    assert_equal(
        blob.group_name,
        metadata_dict["group_name"],
    )
    assert_equal(
        blob.hyperparams,
        metadata.hyperparams,
    )
    assert_equal(
        blob.hyperparams,
        metadata_dict["hyperparams"],
    )
    assert_equal(
        save_load_version,
        metadata.save_load_version,
    )
    assert_equal(
        save_load_version,
        metadata_dict["save_load_version"],
    )
    assert_equal(
        num_shards,
        metadata.num_shards,
    )
    assert_equal(
        num_shards,
        metadata_dict["num_shards"],
    )


def assert_datablob_dataframes_are_equal(
    blob1: sp.DataFrameDataBlob, blob2: sp.DataFrameDataBlob
):
    """
    Assert that the :py:class:`pandas.DataFrame` s in two
    :py:class:`DataBlobDataFrame` s are equal.
    """
    assert_dataframes_are_equal(blob1.set_dataframe(), blob2.set_dataframe())
    assert_dataframes_are_equal(
        blob1.set_training_dataframe(), blob2.set_training_dataframe()
    )
    assert_dataframes_are_equal(
        blob1.set_validation_dataframe(), blob2.set_validation_dataframe()
    )
    assert_dataframes_are_equal(blob1.set_test_dataframe(), blob2.set_test_dataframe())

    assert_dataframes_are_equal(blob1.dataframe, blob2.dataframe)
    assert_dataframes_are_equal(blob1.training_dataframe, blob2.training_dataframe)
    assert_dataframes_are_equal(blob1.validation_dataframe, blob2.validation_dataframe)
    assert_dataframes_are_equal(blob1.test_dataframe, blob2.test_dataframe)


def assert_datablobs_tfdatas_are_equal(blob1: sp.DataBlob, blob2: sp.DataBlob):
    """Assert  that two :py:class:`DataBlob` s are equal."""
    assert_tfdatas_are_equal(blob1.set_training(), blob2.set_training())
    assert_tfdatas_are_equal(blob1.set_validation(), blob2.set_validation())
    assert_tfdatas_are_equal(blob1.set_test(), blob2.set_test())

    assert_tfdatas_are_equal(blob1.training, blob2.training)
    assert_tfdatas_are_equal(blob1.validation, blob2.validation)
    assert_tfdatas_are_equal(blob1.test, blob2.test)


def assert_keras_weights_are_equal(weight_lst_1, weight_lst_2):
    """Assert that two lists of Keras weights are (almost) equal."""
    for w1, w2 in zip(weight_lst_1, weight_lst_2):
        np.testing.assert_almost_equal(w1.numpy(), w2.numpy())


def assert_keras_histories_are_equal(hist1, hist2):
    """Assert that two Keras histories are almost equal."""
    assert_equal(hist1.keys(), hist2.keys())
    for metric in hist1.keys():
        np.testing.assert_almost_equal(hist1[metric], hist2[metric])


def assert_spkeras_models_are_equal(model1, model2):
    """Asserts the approximate equality of :py:class:`~scalarstop.model.KerasModel` classes."""
    assert_equal(model1.name, model2.name)
    assert_equal(model1.current_epoch, model2.current_epoch)
    assert_keras_weights_are_equal(model1.model.weights, model2.model.weights)
    assert_keras_histories_are_equal(model1.history, model2.history)


def assert_model_after_fit(*, return_value, model, expected_epochs: int):
    """Assert that the model's history reflects the epochs that have been trained."""
    assert_equal(return_value, model.history)
    assert_equal(model.current_epoch, expected_epochs)
    for metrics in return_value.values():
        assert_equal(len(metrics), expected_epochs)


def assert_keras_saved_model_directory(model_path):
    """Assert the directory structure of a Keras SavedModel."""
    # Newer versions of TensorFlow might include the file
    # `keras_model.pb`. We want the assertion to pass whether
    # this file is included or not.
    try:
        assert_directory(
            model_path, ["assets", "history.json", "saved_model.pb", "variables"]
        )
    except AssertionError:
        assert_directory(
            model_path,
            [
                "assets",
                "history.json",
                "keras_metadata.pb",
                "saved_model.pb",
                "variables",
            ],
        )
