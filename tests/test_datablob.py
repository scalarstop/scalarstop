"""
Test the scalarstop.datablob module
"""
import doctest
import itertools
import json
import os
import tempfile
import unittest
from typing import Any, Mapping, Optional, Union

import pandas as pd
import tensorflow as tf

import scalarstop as sp
from scalarstop._constants import _DEFAULT_SAVE_LOAD_VERSION
from scalarstop._filesystem import rmtree
from tests.assertions import (
    assert_datablob_dataframes_are_equal,
    assert_datablob_metadatas_are_equal,
    assert_datablobs_tfdatas_are_equal,
    assert_dataframes_are_equal,
    assert_directory,
    assert_hyperparams_are_equal,
    assert_hyperparams_flat_are_equal,
    assert_tfdatas_are_equal,
    tfdata_as_list,
    tfdata_get_first_shape_len,
)


def load_tests(loader, tests, ignore):  # pylint: disable=unused-argument
    """Have the unittest loader also run doctests."""
    tests.addTests(doctest.DocTestSuite(sp.datablob))
    return tests


class MyDataBlob(sp.DataBlob):
    """A simple example of a DataBlob."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams for MyDataBlob,"""

        a: int = 1
        b: str = "hi"

    def __init__(self, *, hyperparams=None, secret: str = "no"):
        """Initialize."""
        super().__init__(hyperparams=hyperparams)
        self._secret = secret

    def set_training(self):
        """Set the training tfdata."""
        return tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5]).map(
            lambda x: x * self.hyperparams.a
        )

    def set_validation(self):
        """Set the validation tfdata."""
        return tf.data.Dataset.from_tensor_slices([6, 7, 8, 9, 10]).map(
            lambda x: x * self.hyperparams.a
        )

    def set_test(self):
        """Set the test tfdata."""
        return tf.data.Dataset.from_tensor_slices([11, 12, 13, 14, 15]).map(
            lambda x: x * self.hyperparams.a
        )


class MyDataBlobArbitraryRows(sp.DataBlob):
    """A DataBlob fixture where we can arbitrarily vary the number of rows."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams for MyDataBlobArbitraryRows."""

        num_training: int
        num_validation: int
        num_test: int
        coefficient: int = 100

    hyperparams: "MyDataBlobArbitraryRows.Hyperparams"

    def _make_tfdata(self, num: int) -> tf.data.Dataset:
        return tf.data.Dataset.range(num).map(
            lambda x: x * self.hyperparams.coefficient
        )

    def set_training(self):
        return self._make_tfdata(self.hyperparams.num_training)

    def set_validation(self):
        return self._make_tfdata(self.hyperparams.num_validation)

    def set_test(self):
        return self._make_tfdata(self.hyperparams.num_test)


class MyDataBlobForgotHyperparams(sp.ModelTemplate):
    """A DataBlob with a misconfigured hyperparams class."""

    Hyperparams = None  # type: ignore


class MyDataBlobRequiredHyperparams(sp.ModelTemplate):
    """A DataBlob where the hyperparams have no default values."""

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams for MyDataBlobRequiredHyperparams."""

        a: int
        b: str


class DataBlobWillFailtoSave(MyDataBlob):
    """
    A :py:class:`DataBlob` that will raise an exception if you try
    to save it to disk.
    """

    def save_hook(self, *, subtype, path) -> None:  # pylint: disable=unused-argument
        """A custom save hook to simulate a failure."""
        # First we check that we have made partial progress in saving
        # this DataBlob.
        this_dataset_path = os.path.dirname(os.path.dirname(path))
        files = os.listdir(this_dataset_path)
        assert len(files) == 1
        assert files[0].startswith(self.name)

        # Then we create an error that should force us to delete this partial progress.
        raise RuntimeError("Simulated failure for testing purposes.")


class DataBlobWillCauseDirectoryNotEmpty(MyDataBlob):
    """
    A :py:class:`DataBlob` designed to fail because a directory
    is created at the exact path that we want to save our
    :py:class:`DataBlob`.
    """

    def save_hook(self, *, subtype, path) -> None:
        """
        We create the destination directory while making the temporary
        directory to simulate a race condition while persisting a
        :py:class:`DataBlob`.
        """
        if subtype == "training":
            datablobs_directory = os.path.dirname(os.path.dirname(path))
            this_dataset_path = os.path.join(datablobs_directory, self.name)
            # We have to create a directory, and then put something inside
            # the directory to make sure that we can't copy into the
            # directory without triggering an error.
            os.mkdir(this_dataset_path)
            with open(
                os.path.join(this_dataset_path, "training"), "w", encoding="utf-8"
            ):
                pass


class DataBlobWillCauseNotADirectoryError(MyDataBlob):
    """
    A :py:class:`DataBlob` designed to fail because a file
    is created at the exact path that we want to save our
    :py:class:`DataBlob`.
    """

    def save_hook(self, *, subtype, path) -> None:
        """
        We create the destination directory as a file, as another
        way to simulate a race condition.
        """
        if subtype == "training":
            datablobs_directory = os.path.dirname(os.path.dirname(path))
            this_dataset_path = os.path.join(datablobs_directory, self.name)
            with open(os.path.join(this_dataset_path), "w", encoding="utf-8"):
                pass


class DataBlobForCaching(sp.DataBlob):
    """
    DataBlob that increments an internal state every time it is
    iterated over.

    The purpose of this is enable caching on this DataBlob and watch
    the counter stop incrementing.
    """

    count = 0

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams for DataBlobForCaching."""

        a: int = 3

    def __init__(self, hyperparams=None):
        """Initialize."""
        super().__init__(hyperparams=hyperparams)

    def _count(self, tensor):
        """Increment the counter for testing."""
        self.count += 1
        return tensor

    def _set_tfdata(self):
        """Generate the tfdata for training, validation, and test."""

        def outer_func(tensor: tf.Tensor) -> tf.Tensor:
            return tf.py_function(self._count, inp=[tensor], Tout=tf.int32)

        return tf.data.Dataset.from_tensor_slices([3, 2, 1]).map(outer_func)

    def set_training(self):
        """Set the training tfdata."""
        return self._set_tfdata()

    def set_validation(self):
        """Set the validation tfdata."""
        return self._set_tfdata()

    def set_test(self):
        """Set the test tfdata."""
        return self._set_tfdata()


class MyDataFrameDataBlob(sp.DataFrameDataBlob):
    """
    An example of creating a :py:class:`DataBlob` with a
    :py:class:`pandas.DataFrame`.
    """

    @sp.dataclass
    class Hyperparams(sp.HyperparamsType):
        """Hyperparams for :py:class:`MyDataFrameDataBlob`."""

        a: int = 0

    def __init__(self, hyperparams=None):
        """Initialize."""
        super().__init__(hyperparams=hyperparams)

    def set_dataframe(self):
        """Set the dataframe."""
        return pd.DataFrame(dict(samples=[1, 2, 3], labels=[4, 5, 6]))

    def transform(self, dataframe: pd.DataFrame):
        """Transform."""
        return tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(dataframe[self.samples_column]),
                tf.data.Dataset.from_tensor_slices(dataframe[self.labels_column]),
            )
        )


class MyAppendDataBlob(sp.AppendDataBlob):
    """Fixture for testing sp.AppendDataBlob."""

    @sp.dataclass
    class Hyperparams(sp.AppendHyperparamsType):
        """Hyperparams for MyAppendDataBlob."""

        coefficient: int

    hyperparams: "MyAppendDataBlob.Hyperparams"

    def __init__(
        self,
        *,
        parent: sp.DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], sp.HyperparamsType]] = None,
        secret2: str,
    ):
        super().__init__(parent=parent, hyperparams=hyperparams)
        self._secret2 = secret2

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.map(
            lambda row: row * self.hyperparams.coefficient,
        )


class MyAppendDataBlobConflictA(sp.AppendDataBlob):
    """
    An AppendDataBlob whose hyperparam `a` is intended to
    conflict with MyDataBlob.
    """

    @sp.dataclass
    class Hyperparams(sp.AppendHyperparamsType):
        """Hyperparams for MyAppendDataBlobConflictA."""

        a: int
        c: int
        d: int

    hyperparams: "MyAppendDataBlobConflictA.Hyperparams"

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        """Multiply the input tf.data.Dataset by our `a` hyperparameter."""
        return tfdata.map(lambda x: x * self.hyperparams.a)


class MyAppendDataBlobConflictB(sp.AppendDataBlob):
    """
    An AppendDataBlob whose hyperparam `b` is intended to
    conflict with MyDataBlob.

    The hyperparam `c` is also meant to conflict with
    MyAppendDataBlobConflictA.
    """

    @sp.dataclass
    class Hyperparams(sp.AppendHyperparamsType):
        """Hyperparams for MyAppendDataBlobConflictB."""

        b: "str"
        c: int
        e: int

    hyperparams: "MyAppendDataBlobConflictB.Hyperparams"

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        """Multiply the input tf.data.Dataset by our `c` hyperparameter."""
        return tfdata.map(lambda x: x * self.hyperparams.c)


class MyAppendDataBlobNoHyperparams(sp.AppendDataBlob):
    """Fixture for testing sp.AppendDataBlob without hyperparams."""

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.enumerate()


class DataBlobTestCase(unittest.TestCase):
    """Base class for unit tests involving DataBlobs."""

    def assert_saved_metadata_json(self, blob, filename):
        """Check that the metadata.json has been properly saved to the filesystem."""
        expected = dict(
            name=blob.name,
            group_name=blob.group_name,
            hyperparams=sp.dataclasses.asdict(blob.hyperparams),
            save_load_version=_DEFAULT_SAVE_LOAD_VERSION,
            num_shards=1,
        )
        with open(filename, "r", encoding="utf-8") as fh:
            actual = json.load(fh)
        self.assertEqual(expected, actual)

    def assert_saved_dataframe(self, blob, subtype, this_datablobs_directory):
        """Check that DataFrames have been properly saved to the filesystem."""
        current_dataframe = getattr(blob, subtype + "_dataframe")
        assert_directory(
            os.path.join(this_datablobs_directory, subtype),
            ["dataframe.pickle.gz", "tfdata", "element_spec.pickle"],
        )
        # Check that the loaded dataframe is the same.
        loaded_dataframe = pd.read_pickle(
            os.path.join(this_datablobs_directory, subtype, "dataframe.pickle.gz")
        )
        assert_dataframes_are_equal(current_dataframe, loaded_dataframe)

    def assertions_for_save(self, blob, datablobs_directory):
        """Assert that saving a DataBlob works."""
        with self.assertRaises(FileExistsError):
            blob.save(datablobs_directory)
        self.assertTrue(os.path.exists(os.path.join(datablobs_directory, blob.name)))
        self.assertTrue(blob.exists_in_datablobs_directory(datablobs_directory))
        this_datablobs_directory = os.path.join(datablobs_directory, blob.name)
        assert_directory(
            this_datablobs_directory,
            ["training", "validation", "test", "metadata.json", "metadata.pickle"],
        )
        self.assert_saved_metadata_json(
            blob, os.path.join(this_datablobs_directory, "metadata.json")
        )
        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                # If the DataBlob has dataframes, check that they have been
                # serialized too.
                current_dataframe = getattr(blob, subtype + "_dataframe", None)
                if current_dataframe is not None:
                    self.assert_saved_dataframe(blob, subtype, this_datablobs_directory)
                else:
                    assert_directory(
                        os.path.join(this_datablobs_directory, subtype),
                        ["tfdata", "element_spec.pickle"],
                    )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(this_datablobs_directory, subtype, "tfdata")
                    )
                )

    def assertions_for_batch_cache_save(self, blob, sequence, datablobs_directory):
        """
        Assert that batching, caching, and saving doesn't change
        names or hyperparams.
        """
        first_name = blob.name
        first_group_name = blob.group_name
        first_hyperparams = blob.hyperparams

        for method_name, kwargs in sequence:
            blob = getattr(blob, method_name)(**kwargs)

        self.assertEqual(blob.name, first_name)
        self.assertEqual(blob.group_name, first_group_name)
        assert_hyperparams_are_equal(blob.hyperparams, first_hyperparams)
        self.assertions_for_save(blob, datablobs_directory)


class TestDataBlob(DataBlobTestCase):
    """Tests for DataBlob."""

    def test_not_implemented(self):
        """
        Test that :py:class:`DataBlob` methods are not implemented
        until overridden.
        """
        blob = sp.DataBlob()
        not_implemented_methods = [
            "set_training",
            "set_validation",
            "set_test",
        ]
        for method_name in not_implemented_methods:
            with self.subTest(method_name + "()"):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(blob, method_name)()
        not_implemented_properties = [
            "training",
            "validation",
            "test",
        ]
        for property_name in not_implemented_properties:
            with self.subTest(property_name):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(blob, property_name)

    def test_names(self):
        """Test that all of the names are correct."""
        blob1 = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
        blob2 = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s2")
        blob3 = MyDataBlob(hyperparams=dict(a=1, b="bye"), secret="s3")
        self.assertTrue(isinstance(blob1, sp.DataBlob))
        self.assertTrue(isinstance(blob2, sp.DataBlob))
        self.assertTrue(isinstance(blob3, sp.DataBlob))
        self.assertEqual(blob1.name, "MyDataBlob-naro6iqyw9whazvkgp4w3qa2")
        self.assertEqual(blob1.name, blob2.name)
        self.assertEqual(blob3.name, "MyDataBlob-cmfhzgfa6z4gm43ntk1q2hbp")
        self.assertEqual(blob1.group_name, "MyDataBlob")
        self.assertEqual(blob1.group_name, blob2.group_name)
        self.assertEqual(blob1.group_name, blob3.group_name)

    def test_missing_hyperparams_class(self):
        """Test what happens when the hyperparams class itself is missing."""
        with self.assertRaises(sp.exceptions.YouForgotTheHyperparams):
            MyDataBlobForgotHyperparams()

    def test_save_success(self):
        """Test that we can save a :py:class:`DataBlob`."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            # Make sure that the DataBlob has not already been saved.
            self.assertFalse(blob.exists_in_datablobs_directory(datablobs_directory))

            # Save the DataBlob and check that it was successfully persisted
            # to the filesystem.
            blob.save(datablobs_directory)
            self.assertTrue(blob.exists_in_datablobs_directory(datablobs_directory))

            # Test that we raise an exception when the datablob already exists.
            with self.assertRaises(sp.exceptions.FileExists):
                blob.save(datablobs_directory)

            # Test that we can suppress the exception that we just raised
            blob.save(datablobs_directory, ignore_existing=True)

            # Test that the saved data looks right.
            self.assertions_for_save(blob, datablobs_directory)

            # Check that serialized element spec is correct.
            for subtype in ["training", "validation", "test"]:
                tfdata = getattr(blob, subtype)
                with open(
                    os.path.join(
                        datablobs_directory, blob.name, subtype, "element_spec.pickle"
                    ),
                    "rb",
                ) as fh:
                    loaded_element_spec = sp.pickle.load(fh)
                self.assertEqual(tfdata.element_spec, loaded_element_spec)

    def test_save_catch_exception(self):
        """
        Test that :py:meth:`DataBlob.save` deletes partially-saved
        data if it fails.
        """
        with tempfile.TemporaryDirectory() as datablobs_directory:
            with self.assertRaises(RuntimeError):
                DataBlobWillFailtoSave().save(datablobs_directory)
            assert_directory(datablobs_directory, [])

    def test_save_dataset_created_during_creation_1(self):
        """
        Test what happens when the final :py:class:`DataBlob`
        directory is created after we start (but do not finish)
        saving our :py:class:`DataBlob`.
        """
        with tempfile.TemporaryDirectory() as datablobs_directory:
            with self.assertRaises(sp.exceptions.FileExistsDuringDataBlobCreation):
                DataBlobWillCauseDirectoryNotEmpty().save(datablobs_directory)

    def test_save_dataset_created_during_creation_2(self):
        """
        Test what happpens when we create a file (not a directory)
        at the location that we wanted to create the directory
        to save our :py:class:`DataBlob`.
        """
        with tempfile.TemporaryDirectory() as datablobs_directory:
            with self.assertRaises(sp.exceptions.FileExistsDuringDataBlobCreation):
                DataBlobWillCauseNotADirectoryError().save(datablobs_directory)

    def test_from_exact_path(self):
        """Test that we can load a :py:class:`DataBlob` from the filesystem."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            blob.save(datablobs_directory)
            self.assertTrue(blob.exists_in_datablobs_directory(datablobs_directory))
            loaded = sp.DataBlob.from_exact_path(
                os.path.join(datablobs_directory, blob.name)
            )
            assert_datablob_metadatas_are_equal(blob, loaded)
            assert_datablobs_tfdatas_are_equal(blob, loaded)

    def test_load_dataset_not_found_1(self):
        """
        Test what happens when we try to load a nonexistent
        :py:class:`DataBlob` from the filesystem.
        """
        with self.assertRaises(sp.exceptions.DataBlobNotFound):
            sp.DataBlob.from_exact_path("asdf")

    def test_load_dataset_not_found_2(self):
        """
        Test what happens when we delete a directory containing
        a :py:class:`tf.data.Dataset` and the element spec.
        """
        for deleted_subtype in ["training", "validation", "test"]:
            with tempfile.TemporaryDirectory() as datablobs_directory:
                blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1").save(
                    datablobs_directory
                )
                rmtree(os.path.join(datablobs_directory, blob.name, deleted_subtype))
                loaded = sp.DataBlob.from_exact_path(
                    os.path.join(datablobs_directory, blob.name)
                )
                for loaded_subtype in ["training", "validation", "test"]:
                    with self.subTest(
                        f"deleted {deleted_subtype}, loaded {loaded_subtype}"
                    ):
                        if deleted_subtype == loaded_subtype:
                            with self.assertRaises(
                                sp.exceptions.TensorFlowDatasetNotFound
                            ):
                                getattr(loaded, loaded_subtype)
                        else:
                            getattr(loaded, loaded_subtype)

    def test_load_dataset_not_found_3(self):
        """
        Test what happens when we delete a directory containing a
        :py:class:`tf.data.Dataset` but we don't delete the element spec.
        """
        for deleted_subtype in ["training", "validation", "test"]:
            with tempfile.TemporaryDirectory() as datablobs_directory:
                blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1").save(
                    datablobs_directory
                )
                rmtree(
                    os.path.join(
                        datablobs_directory, blob.name, deleted_subtype, "tfdata"
                    )
                )
                loaded = sp.DataBlob.from_exact_path(
                    os.path.join(datablobs_directory, blob.name)
                )
                for loaded_subtype in ["training", "validation", "test"]:
                    with self.subTest(
                        f"deleted {deleted_subtype}, loaded {loaded_subtype}"
                    ):
                        if deleted_subtype == loaded_subtype:
                            with self.assertRaises(
                                sp.exceptions.TensorFlowDatasetNotFound
                            ):
                                getattr(loaded, loaded_subtype)
                        else:
                            getattr(loaded, loaded_subtype)

    def test_cache_save_load_permutations(self):
        """Test loading the dataset after cache and or save."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            operations = dict(
                cache={},
                save=dict(datablobs_directory=datablobs_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataBlob(hyperparams=dict(a=idx, b="hi"), secret="s1")
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, datablobs_directory
                    )
                    loaded = blob.from_exact_path(
                        os.path.join(datablobs_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablobs_tfdatas_are_equal(blob, loaded)

    def test_batch_cache_save_load_permutations(self):
        """Test loading the dataset after batch/cache/save."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            operations = dict(
                batch=dict(batch_size=2),
                cache={},
                save=dict(datablobs_directory=datablobs_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataBlob(hyperparams=dict(a=idx, b="hi"), secret="s1")
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, datablobs_directory
                    )
                    loaded = blob.from_exact_path(
                        os.path.join(datablobs_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)


class Test_WrapDataBlob(DataBlobTestCase):
    """Test the :py:class:`_WrapDataBlob` class."""

    def test_not_implemented(self):
        """
        Test that :py:class:`_WrapDataBlob` methods are not
        implemented until overridden.
        """
        wrapped = sp.datablob._WrapDataBlob(
            wraps=MyDataBlob(), training=True, validation=True, test=True
        )
        not_implemented_methods = [
            "set_training",
            "set_validation",
            "set_test",
        ]
        for method_name in not_implemented_methods:
            with self.subTest(method_name + "()"):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(wrapped, method_name)()
        not_implemented_properties = [
            "training",
            "validation",
            "test",
        ]
        for property_name in not_implemented_properties:
            with self.subTest(property_name):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(wrapped, property_name)

    def test_cache_wrapping(self):
        """
        Test that `_WrapDataBlob.training` calls `parent.training` a
        opposed to `parent.set_training()`.
        """

        class _TrainingCalled(Exception):
            """Test that DataBlob.training is called."""

        class _ValidationCalled(Exception):
            """Test that DataBlob.validation is called."""

        class _TestCalled(Exception):
            """Test that DataBlob.test is called."""

        class _Parent(MyDataBlob):
            """Parent class with instrumented training, validation, and test property() methods."""

            @property
            def training(self):
                raise _TrainingCalled("training")

            @property
            def validation(self):
                raise _ValidationCalled("validation")

            @property
            def test(self):
                raise _TestCalled("test")

        wrapped = sp.datablob._WrapDataBlob(
            wraps=_Parent(),
            training=True,
            validation=True,
            test=True,
        )
        with self.assertRaises(_TrainingCalled):
            wrapped.training  # pylint: disable=pointless-statement
        with self.assertRaises(_ValidationCalled):
            wrapped.validation  # pylint: disable=pointless-statement
        with self.assertRaises(_TestCalled):
            wrapped.test  # pylint: disable=pointless-statement

    def test_disable_training_validation_test(self):
        """
        Test that _WrapDataBlob can selectively disable the training, validation, and test suites.
        """

        class _Wrapper(sp.datablob._WrapDataBlob):
            def _wrap_tfdata(self, tfdata):
                return tfdata.map(lambda x: x * 1000)

        parent = MyDataBlob()
        wrapped_training = _Wrapper(
            wraps=parent,
            training=True,
            validation=False,
            test=False,
        )
        expected_training = tf.data.Dataset.from_tensor_slices(
            [1000, 2000, 3000, 4000, 5000]
        )
        assert_tfdatas_are_equal(expected_training, wrapped_training.training)
        assert_tfdatas_are_equal(parent.validation, wrapped_training.validation)
        assert_tfdatas_are_equal(parent.test, wrapped_training.test)

        wrapped_validation = _Wrapper(
            wraps=parent,
            training=False,
            validation=True,
            test=False,
        )
        expected_validation = tf.data.Dataset.from_tensor_slices(
            [6000, 7000, 8000, 9000, 10000]
        )
        assert_tfdatas_are_equal(parent.training, wrapped_validation.training)
        assert_tfdatas_are_equal(expected_validation, wrapped_validation.validation)
        assert_tfdatas_are_equal(parent.test, wrapped_validation.test)

        wrapped_test = _Wrapper(
            wraps=parent,
            training=False,
            validation=False,
            test=True,
        )
        expected_test = tf.data.Dataset.from_tensor_slices(
            [11000, 12000, 13000, 14000, 15000]
        )
        assert_tfdatas_are_equal(parent.training, wrapped_test.training)
        assert_tfdatas_are_equal(parent.validation, wrapped_test.validation)
        assert_tfdatas_are_equal(expected_test, wrapped_test.test)


class Test_BatchDataBlob(DataBlobTestCase):
    """Tests for _BatchDataBlob"""

    def test_successs(self):
        """Test that _BatchDataBlob works."""
        blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
        batched = blob.batch(2)
        self.assertTrue(isinstance(blob, sp.DataBlob))
        self.assertTrue(isinstance(batched, sp.DataBlob))
        self.assertEqual(blob.name, batched.name)
        self.assertEqual(blob.group_name, batched.group_name)
        self.assertEqual(blob.hyperparams, batched.hyperparams)

        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                lst = list(getattr(batched, subtype))
                self.assertEqual(len(lst), 3)
                self.assertEqual(lst[0].shape, (2,))
                self.assertEqual(lst[1].shape, (2,))
                self.assertEqual(lst[2].shape, (1,))

    def test_batch_with_tensorflow_distribute(self):
        """Test batching with the default TensorFlow Distribute strategy."""
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        input_batch_size = 2
        batched = MyDataBlob().batch(input_batch_size, with_tf_distribute=True)
        self.assertEqual(num_replicas * input_batch_size, batched.batch_size)

    def test_selectively_disabling_batch(self):
        """Test that training/validation/test=False disables batching."""
        datablob = MyDataBlob()
        self.assertEqual(tfdata_get_first_shape_len(datablob.training), 0)
        self.assertEqual(tfdata_get_first_shape_len(datablob.validation), 0)
        self.assertEqual(tfdata_get_first_shape_len(datablob.test), 0)

        batched_training = datablob.batch(2, validation=False, test=False)
        self.assertEqual(tfdata_get_first_shape_len(batched_training.training), 1)
        self.assertEqual(tfdata_get_first_shape_len(batched_training.validation), 0)
        self.assertEqual(tfdata_get_first_shape_len(batched_training.test), 0)

        batched_training_and_val = batched_training.batch(2, test=False)
        self.assertEqual(
            tfdata_get_first_shape_len(batched_training_and_val.training), 2
        )
        self.assertEqual(
            tfdata_get_first_shape_len(batched_training_and_val.validation), 1
        )
        self.assertEqual(tfdata_get_first_shape_len(batched_training_and_val.test), 0)

        batched_val_and_test = batched_training_and_val.batch(2, training=False)
        self.assertEqual(tfdata_get_first_shape_len(batched_val_and_test.training), 2)
        self.assertEqual(tfdata_get_first_shape_len(batched_val_and_test.validation), 2)
        self.assertEqual(tfdata_get_first_shape_len(batched_val_and_test.test), 1)


class Test_CacheDataBlob(DataBlobTestCase):
    """Tests for _CacheDataBlob."""

    def test_success(self):
        """Test that in-memory caching works."""
        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                # Each iteration increments the count by 3 because the tf.data pipeline is
                # not cached.
                blob = DataBlobForCaching()

                # We start at 0.
                self.assertEqual(blob.count, 0)
                # Generating the tf.data pipeline does not increment the count.
                subtype_tf = getattr(blob, subtype)
                self.assertEqual(blob.count, 0)

                # Each time we iterate over the tf.data pipeline, the count
                # value goes up by 3. The counter should stop incrementing
                # once we begin caching the pipeline.
                for _ in subtype_tf:
                    continue
                self.assertEqual(blob.count, 3)
                for _ in subtype_tf:
                    continue
                self.assertEqual(blob.count, 6)
                for _ in subtype_tf:
                    continue
                self.assertEqual(blob.count, 9)

                # Set up the cached pipeline and verify everything is the same.
                cached = blob.cache()
                self.assertEqual(blob.name, cached.name)
                self.assertEqual(blob.group_name, cached.group_name)
                self.assertEqual(blob.hyperparams, cached.hyperparams)

                # The count is where we last left it.
                self.assertEqual(cached.count, 9)

                # Selecting a tf.data pipeline from our DataBlob does not trigger
                # an iteration over the pipeline. This means that the count is still 9.
                cached_subtype_tf = getattr(cached, subtype)
                self.assertEqual(cached.count, 9)

                # The first iteration over a cached tf.data pipeline will still
                # increment the counter. This is because tf.data caching isn't complete
                # until the next pass over the entire dataset.
                for _ in cached_subtype_tf:
                    continue
                self.assertEqual(cached.count, 12)

                # Now that caching is complete, we expect the value to stay at 12.
                for _ in cached_subtype_tf:
                    continue
                self.assertEqual(cached.count, 12)
                for _ in cached_subtype_tf:
                    continue
                self.assertEqual(cached.count, 12)

    def test_selectively_disable_caching(self):
        """Test that training=False disables caching on the training set, and so forth."""
        cached = DataBlobForCaching().cache(
            training=False, precache_validation=True, precache_test=True
        )
        # the count is 6 because precaching the validation and test sets incremented
        # the counter.
        self.assertEqual(cached.count, 6)

        # The training set is not cached, so our count increments by another 3.
        for _ in cached.training:
            continue
        self.assertEqual(cached.count, 9)

        # The validation and test sets have been precached,
        # so they will no longer increment the count.
        for _ in cached.validation:
            continue
        self.assertEqual(cached.count, 9)
        for _ in cached.test:
            continue
        self.assertEqual(cached.count, 9)
        for _ in cached.validation:
            continue
        self.assertEqual(cached.count, 9)
        for _ in cached.test:
            continue
        self.assertEqual(cached.count, 9)

        # But because we have not cached the training set, the count
        # will continue to increment every time we iterate over iut.
        for _ in cached.training:
            continue
        self.assertEqual(cached.count, 12)
        for _ in cached.training:
            continue
        self.assertEqual(cached.count, 15)

    def test_precache_training(self):
        """Test CacheDataBlob precache_training=True."""
        # Each iteration increments the count by 3 because the tf.data pipeline is not cached.
        blob = DataBlobForCaching()

        # We start at 0.
        self.assertEqual(blob.count, 0)

        # Generating the tf.data pipeline does not increment the count.
        subtype_tf = getattr(blob, "training")
        self.assertEqual(blob.count, 0)

        # Each time we iterate over the tf.data pipeline, the count
        # value goes up by 3. The counter should stop incrementing
        # once we begin caching the pipeline.
        for _ in subtype_tf:
            continue
        self.assertEqual(blob.count, 3)
        for _ in subtype_tf:
            continue
        self.assertEqual(blob.count, 6)
        for _ in subtype_tf:
            continue
        self.assertEqual(blob.count, 9)

        # Set up the cached pipeline and verify everything is the same.
        cached = blob.cache(
            precache_training=True, precache_validation=False, precache_test=False
        )
        self.assertEqual(blob.name, cached.name)
        self.assertEqual(blob.group_name, cached.group_name)
        self.assertEqual(blob.hyperparams, cached.hyperparams)

        # The count is where we last left it.
        self.assertEqual(cached.count, 12)

        # Selecting a tf.data pipeline from our DataBlob does not trigger
        # an iteration over the pipeline. This means that the count is still 9.
        cached_subtype_tf = getattr(cached, "training")
        self.assertEqual(cached.count, 12)

        # The first iteration over a cached tf.data pipeline will still
        # increment the counter. This is because tf.data caching isn't complete
        # until the next pass over the entire dataset.
        for _ in cached_subtype_tf:
            continue
        self.assertEqual(cached.count, 12)

        # Now that caching is complete, we expect the value to stay at 12.
        for _ in cached_subtype_tf:
            continue
        self.assertEqual(cached.count, 12)
        for _ in cached_subtype_tf:
            continue
        self.assertEqual(cached.count, 12)

    def test_precache_all(self):
        """Test CacheDataBlob precache True for training/validation/test."""
        # Each iteration increments the count by 3 because the tf.data pipeline is not cached.
        blob = DataBlobForCaching()
        self.assertEqual(blob.count, 0)
        for subtype in ["training", "validation", "test"]:
            subtype_tf = getattr(blob, subtype)
            for _ in subtype_tf:
                continue
            for _ in subtype_tf:
                continue
        self.assertEqual(blob.count, 18)
        # Set up the cached pipeline and verify everything is the same.
        cached = blob.cache(
            precache_training=True, precache_validation=True, precache_test=True
        )
        assert_datablob_metadatas_are_equal(blob, cached)
        # The training, validation, and test sets added 9 each. 9 + 18 brings us to 27.
        self.assertEqual(cached.count, 27)
        for subtype in ["training", "validation", "test"]:
            cached_subtype_tf = getattr(cached, subtype)
            # Now that caching is complete, we expect the value to stay at 27.
            for _ in cached_subtype_tf:
                continue
            self.assertEqual(cached.count, 27)
            for _ in cached_subtype_tf:
                continue
            self.assertEqual(cached.count, 27)

    def test_inconsistent_caching_parameters(self):
        """Test that we raise an exception when caching parameters don't make sense."""
        blob = DataBlobForCaching()
        with self.assertRaises(sp.exceptions.InconsistentCachingParameters):
            blob.cache(training=False, precache_training=True)
        with self.assertRaises(sp.exceptions.InconsistentCachingParameters):
            blob.cache(validation=False, precache_validation=True)
        with self.assertRaises(sp.exceptions.InconsistentCachingParameters):
            blob.cache(test=False, precache_test=True)


class Test_WithOptionsDataBlob(DataBlobTestCase):
    """Tests for _WithOptionsDataBlob."""

    def setUp(self):
        self.tf_options = tf.data.Options()
        self.tf_options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

    def test_success(self):
        """Test that setting options works."""
        blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
        # Check that the auto sharding polciy is AUTO by default.
        for subtype in ["training", "validation", "test"]:
            tfdata = getattr(blob, subtype)
            self.assertEqual(
                tfdata.options().experimental_distribute.auto_shard_policy,
                tf.data.experimental.AutoShardPolicy.AUTO,
            )
        # Try setting the AutoShardPolicy to DATA.
        blob_with_options = blob.with_options(self.tf_options)

        assert_hyperparams_are_equal(blob.hyperparams, blob_with_options.hyperparams)
        assert_hyperparams_flat_are_equal(
            blob.hyperparams_flat, blob_with_options.hyperparams_flat
        )

        # Check that the sharding policy has been properly applied.
        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                tfdata = getattr(blob_with_options, subtype)
                self.assertEqual(
                    tfdata.options().experimental_distribute.auto_shard_policy,
                    tf.data.experimental.AutoShardPolicy.DATA,
                )

    def test_selectively_disabling_options(self):
        """test that training/validation/test=False disables setting tf.data.Options."""
        # Let's try setting optiosn on the training and test sets,
        # but not the validation set.
        AUTO = tf.data.experimental.AutoShardPolicy.AUTO
        DATA = tf.data.experimental.AutoShardPolicy.DATA
        blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
        blob_with_options = blob.with_options(
            self.tf_options,
            validation=False,
        )
        assert_hyperparams_are_equal(blob.hyperparams, blob_with_options.hyperparams)
        assert_hyperparams_flat_are_equal(
            blob.hyperparams_flat, blob_with_options.hyperparams_flat
        )
        self.assertEqual(
            blob_with_options.training.options().experimental_distribute.auto_shard_policy,
            DATA,
        )
        self.assertEqual(
            blob_with_options.validation.options().experimental_distribute.auto_shard_policy,
            AUTO,
        )
        self.assertEqual(
            blob_with_options.test.options().experimental_distribute.auto_shard_policy,
            DATA,
        )


class TestDataFrameDataBlob(DataBlobTestCase):
    """Tests for DataFrameDataBlob."""

    def test_not_implemented(self):
        """
        Test that :py:class:`DataFrameDataBlob` methods are not
        implemented until overridden.
        """
        blob = sp.DataFrameDataBlob()
        not_implemented_methods = [
            "set_dataframe",
            "set_training_dataframe",
            "set_validation_dataframe",
            "set_test_dataframe",
            "set_training",
            "set_validation",
            "set_test",
        ]
        for method_name in not_implemented_methods:
            with self.subTest(method_name + "()"):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(blob, method_name)()

        with self.subTest("transform()"):
            with self.assertRaises(sp.exceptions.IsNotImplemented):
                blob.transform(pd.DataFrame(dict(a=[1, 2], b=[3, 4])))

        not_implemented_properties = [
            "training_dataframe",
            "validation_dataframe",
            "test_dataframe",
            "training",
            "validation",
            "test",
        ]
        for property_name in not_implemented_properties:
            with self.subTest(property_name):
                with self.assertRaises(sp.exceptions.IsNotImplemented):
                    getattr(blob, property_name)

    def test_save(self):
        """Test that we can save a DataFrameDataBlob."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            blob = MyDataFrameDataBlob()
            self.assertFalse(blob.exists_in_datablobs_directory(datablobs_directory))
            blob.save(datablobs_directory)
            self.assertTrue(blob.exists_in_datablobs_directory(datablobs_directory))

    def test_from_exact_path(self):
        """
        Test that we can load a :py:class:`DataFrameDataBlob`
        from the filesystem.
        """
        with tempfile.TemporaryDirectory() as datablobs_directory:
            blob = MyDataFrameDataBlob()
            self.assertFalse(blob.exists_in_datablobs_directory(datablobs_directory))
            blob.save(datablobs_directory)
            self.assertTrue(blob.exists_in_datablobs_directory(datablobs_directory))
            loaded = MyDataFrameDataBlob.from_exact_path(
                os.path.join(datablobs_directory, blob.name)
            )
            assert_datablob_metadatas_are_equal(blob, loaded)
            assert_datablobs_tfdatas_are_equal(blob, loaded)
            assert_datablob_dataframes_are_equal(blob, loaded)

    def test_cache_save_load_permutations(self):
        """Test loading the dataset after cache and or save."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            operations = dict(
                cache={},
                save=dict(datablobs_directory=datablobs_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataFrameDataBlob(hyperparams=dict(a=idx))
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, datablobs_directory
                    )
                    loaded = blob.from_exact_path(
                        os.path.join(datablobs_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablobs_tfdatas_are_equal(blob, loaded)
                    assert_datablob_dataframes_are_equal(blob, loaded)

    def test_batch_cache_save_load_permutations(self):
        """Test loading the dataset after batch/cache/save."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            operations = dict(
                batch=dict(batch_size=2),
                cache={},
                save=dict(datablobs_directory=datablobs_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataFrameDataBlob(hyperparams=dict(a=idx))
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, datablobs_directory
                    )
                    loaded = blob.from_exact_path(
                        os.path.join(datablobs_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablob_dataframes_are_equal(blob, loaded)


class TestAppendDataBlob(unittest.TestCase):
    """Tests for AppendDataBlob."""

    def assert_parentage(self, *, parent, append):
        """A basket of assertions for parent and child DataBlobs."""
        # Check append.parent points to parent.
        assert_datablob_metadatas_are_equal(parent, append.parent)
        assert_datablobs_tfdatas_are_equal(parent, append.parent)

        # Check that append.hyperparams contains the parent's hyperparams.
        self.assertEqual(parent.name, append.hyperparams.parent.name)
        self.assertEqual(parent.group_name, append.hyperparams.parent.group_name)

        # Check that we have embedded the parent's hyperparams.
        assert_hyperparams_are_equal(parent.hyperparams, append.parent.hyperparams)

        # Check that our AppendDataset has a unique name
        # that is different from the parent.
        self.assertTrue(append.name != parent.name)
        self.assertTrue(append.group_name != parent.group_name)

    def test_not_implemented(self):
        """
        Test that AppendDataBlob's _wrap_tfdata() is not implemented.

        We have to implement that method in a subclass in order
        for it to work.
        """
        parent = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
        append = sp.AppendDataBlob(parent=parent)
        self.assert_parentage(parent=parent, append=append)
        assert_hyperparams_flat_are_equal(
            parent.hyperparams_flat, append.hyperparams_flat
        )

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            append.training  # pylint: disable=pointless-statement
        with self.assertRaises(sp.exceptions.IsNotImplemented):
            append.validation  # pylint: disable=pointless-statement
        with self.assertRaises(sp.exceptions.IsNotImplemented):
            append.test  # pylint: disable=pointless-statement

    def test_success(self):
        """Test that AppendDataBlob works."""
        coefficient = 10
        append_hyperparams = dict(coefficient=coefficient)
        parent_hyperparams = dict(a=1, b="hi")
        parent = MyDataBlob(hyperparams=parent_hyperparams, secret="s1")
        append = MyAppendDataBlob(
            parent=parent, hyperparams=append_hyperparams, secret2="secret2"
        )
        # Assert the basics.
        self.assert_parentage(parent=parent, append=append)
        assert_hyperparams_flat_are_equal(
            dict(**parent_hyperparams, **append_hyperparams), append.hyperparams_flat
        )
        self.assertEqual(
            set(sp.enforce_dict(append.hyperparams)), set(["parent", "coefficient"])
        )
        # Check that the transformation specified in _wrap_tfdata()
        # actually happened.
        for subtype in ("training", "validation", "test"):
            with self.subTest(subtype):
                for parent_tensor, append_tensor in zip(
                    getattr(parent, subtype), getattr(append, subtype)
                ):
                    parent_value = parent_tensor.numpy()
                    append_value = append_tensor.numpy()
                    self.assertEqual(parent_value * coefficient, append_value)

    def test_success_no_additional_hyperparams(self):
        """Test that AppendDataBlob works when we have no additional hyperparams to add."""
        parent_hyperparams = dict(a=1, b="hi")
        parent = MyDataBlob(hyperparams=parent_hyperparams, secret="s1")
        append = MyAppendDataBlobNoHyperparams(parent=parent)

        # Assert the basics.
        self.assert_parentage(parent=parent, append=append)
        assert_hyperparams_flat_are_equal(
            parent.hyperparams_flat, append.hyperparams_flat
        )

        # Assert that we have no other hyperparams.
        self.assertEqual(set(sp.enforce_dict(append.hyperparams)), set(["parent"]))

        # Check that the transformation specified in _wrap_tfdata()
        # actually happened.
        for subtype in ("training", "validation", "test"):
            with self.subTest(subtype):
                for idx, (
                    parent_tensor,
                    (append_tensor_idx_tensor, append_tensor),
                ) in enumerate(zip(getattr(parent, subtype), getattr(append, subtype))):
                    parent_value = parent_tensor.numpy()
                    append_tensor_idx = append_tensor_idx_tensor.numpy()
                    append_value = append_tensor.numpy()
                    self.assertEqual(idx, append_tensor_idx)
                    self.assertEqual(parent_value, append_value)

    def test_with_dataframe(self):
        """Test that an AppendDataBlob can inherit from a DataFrameDataBlob."""
        parent = MyDataFrameDataBlob()
        append = MyAppendDataBlobNoHyperparams(parent=parent)

        # Assert the basics.
        self.assert_parentage(parent=parent, append=append)

        # Assert that the dataframes are the same.
        assert_datablob_dataframes_are_equal(parent, append)

    def test_batch(self):
        """Test batching on AppendDataBlob."""
        append_hyperparams = dict(coefficient=10)
        parent_hyperparams = dict(a=1, b="hi")
        parent = MyDataBlob(hyperparams=parent_hyperparams, secret="s1")
        append = MyAppendDataBlob(
            parent=parent,
            hyperparams=append_hyperparams,
            secret2="secret2",
        )
        self.assert_parentage(parent=parent, append=append)
        batched = append.batch(2)
        self.assert_parentage(parent=parent, append=batched)
        assert_hyperparams_flat_are_equal(
            dict(**parent_hyperparams, **append_hyperparams), batched.hyperparams_flat
        )
        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                lst = list(getattr(batched, subtype))
                self.assertEqual(len(lst), 3)
                self.assertEqual(lst[0].shape, (2,))
                self.assertEqual(lst[1].shape, (2,))
                self.assertEqual(lst[2].shape, (1,))

    def test_save_and_from_exact_path(self):
        """Test that we can save an AppendDatablob and load it back."""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            coefficient = 10
            parent = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            append = MyAppendDataBlob(
                parent=parent,
                hyperparams=dict(coefficient=coefficient),
                secret2="secret2",
            )
            append.save(datablobs_directory)
            self.assertTrue(append.exists_in_datablobs_directory(datablobs_directory))
            self.assertFalse(parent.exists_in_datablobs_directory(datablobs_directory))
            append_loaded = MyAppendDataBlob.from_exact_path(
                os.path.join(datablobs_directory, append.name)
            )
            assert_datablob_metadatas_are_equal(append, append_loaded)
            assert_datablobs_tfdatas_are_equal(append, append_loaded)

    def test_from_filesystem_with_parent(self):
        """Test AppendDataBlob.from_filesystem_with_parent()"""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            coefficient = 10
            parent = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            append = MyAppendDataBlob(
                parent=parent,
                hyperparams=dict(coefficient=coefficient),
                secret2="secret2",
            )
            append.save(datablobs_directory)
            append_loaded = MyAppendDataBlob.from_filesystem_with_parent(
                parent=parent,
                hyperparams=dict(coefficient=coefficient),
                datablobs_directory=datablobs_directory,
            )
            assert_datablob_metadatas_are_equal(append, append_loaded)
            assert_datablobs_tfdatas_are_equal(append, append_loaded)

    def test_from_filesystem_or_new_with_parent(self):
        """Test AppendDataBlob.from_filesystem_or_new_with_parent()"""
        with tempfile.TemporaryDirectory() as datablobs_directory:
            coefficient = 10
            parent = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            append = MyAppendDataBlob.from_filesystem_or_new_with_parent(
                parent=parent,
                hyperparams=dict(coefficient=coefficient),
                secret2="secret2",
                datablobs_directory=datablobs_directory,
            )
            append.save(datablobs_directory)
            append_loaded = MyAppendDataBlob.from_filesystem_or_new_with_parent(
                parent=parent,
                hyperparams=dict(coefficient=coefficient),
                secret2="secret2",
                datablobs_directory=datablobs_directory,
            )
            assert_datablob_metadatas_are_equal(append, append_loaded)
            assert_datablobs_tfdatas_are_equal(append, append_loaded)

    def test_hyperparams_flat_works(self):
        """
        Assert that the hyperparams_flat() property flattens
        nested hyperparams from AppendDataBlobs.
        """
        parent = MyDataBlob(hyperparams=dict(a=2, b="hi"), secret="s1")
        child = MyAppendDataBlobConflictA(
            parent=parent, hyperparams=dict(a=5, c=3, d=4)
        )
        grandchild = MyAppendDataBlobConflictB(
            parent=child, hyperparams=dict(b="lol", c=6, e=7)
        )

        self.assert_parentage(parent=parent, append=child)
        self.assert_parentage(parent=child, append=grandchild)
        self.assert_parentage(parent=parent, append=grandchild.parent)

        self.assertEqual(
            grandchild.hyperparams.parent.hyperparams.parent.name, parent.name
        )
        self.assertEqual(
            grandchild.hyperparams.parent.hyperparams.parent.group_name,
            parent.group_name,
        )
        assert_hyperparams_are_equal(
            grandchild.hyperparams.parent.hyperparams.parent.hyperparams,
            parent.hyperparams,
        )

        self.assertEqual(parent.hyperparams_flat, dict(a=2, b="hi"))
        self.assertEqual(child.hyperparams_flat, dict(a=5, b="hi", c=3, d=4))
        self.assertEqual(grandchild.hyperparams_flat, dict(a=5, b="lol", d=4, c=6, e=7))

        self.assertEqual([2, 4, 6, 8, 10], tfdata_as_list(parent.training))
        self.assertEqual([10, 20, 30, 40, 50], tfdata_as_list(child.training))
        self.assertEqual([60, 120, 180, 240, 300], tfdata_as_list(grandchild.training))


class TestDataBlobSaveLoadVersions(unittest.TestCase):
    """
    Test the ``load_save_version`` part of DataBlob.

    We want to make sure that we can version the protocol
    that we use to save and load DataBlobs to/from disk.
    Ideally, we want ScalarStop to be both backwards-compatible
    and forwards-compatible.
    """

    @classmethod
    def setUpClass(cls):
        cls.datablob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")

    def setUp(self):
        self._datablobs_directory_handle = (
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )
        self.datablobs_directory = self._datablobs_directory_handle.name

    def tearDown(self):
        self._datablobs_directory_handle.cleanup()

    def test_save_v1_does_not_support_sharding(self):
        """Test that the ``load_save_version`` 1 does not support sharding."""
        with self.assertRaises(sp.exceptions.DataBlobShardingValueError):
            self.datablob.save(
                datablobs_directory=self.datablobs_directory,
                save_load_version=1,
                num_shards=2,
            )
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )

    def test_load_fails_with_shard_offset_too_high(self):
        """Test that we cannot load a DataBlob when shard_offset >= num_shards."""
        self.datablob.save(
            datablobs_directory=self.datablobs_directory,
            num_shards=2,
        )
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory,
            shard_offset=3,
        )
        with self.assertRaises(sp.exceptions.DataBlobShardingValueError):
            loaded.training  # pylint: disable=pointless-statement

    def test_load_fails_with_shard_quantity_too_high(self):
        """Test that we cannot load a DataBlob when shard_quantity >= num_shards."""
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        self.datablob.save(
            datablobs_directory=self.datablobs_directory,
            num_shards=2,
        )
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory,
            shard_offset=0,
            shard_quantity=4,
        )
        with self.assertRaises(sp.exceptions.DataBlobShardingValueError):
            loaded.training  # pylint: disable=pointless-statement

    def test_load_fails_with_shard_num_shards_and_quantity_too_high(self):
        """Test that we cannot load a DataBlob when shard_offset + shard_quantity >= num_shards."""
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        self.datablob.save(
            datablobs_directory=self.datablobs_directory,
            num_shards=2,
        )
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory,
            shard_offset=2,
            shard_quantity=2,
        )
        with self.assertRaises(sp.exceptions.DataBlobShardingValueError):
            loaded.training  # pylint: disable=pointless-statement

    def test_save_vdefault_and_load(self):
        """Test that we can load and save datablobs with the default protocol."""
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        self.datablob.save(datablobs_directory=self.datablobs_directory)
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory
        )
        assert_datablob_metadatas_are_equal(self.datablob, loaded)
        assert_datablobs_tfdatas_are_equal(self.datablob, loaded)

    def test_save_v1_and_load(self):
        """Test that we can save and load datablobs with protocol v1."""
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        self.datablob.save(
            datablobs_directory=self.datablobs_directory, save_load_version=1
        )
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory
        )
        assert_datablob_metadatas_are_equal(self.datablob, loaded)
        assert_datablobs_tfdatas_are_equal(self.datablob, loaded)

    def test_save_v2_and_load(self):
        """Test that we can save and load datablobs with protocol v2."""
        self.assertFalse(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        self.datablob.save(
            datablobs_directory=self.datablobs_directory,
            save_load_version=2,
        )
        self.assertTrue(
            self.datablob.exists_in_datablobs_directory(self.datablobs_directory)
        )
        loaded = MyDataBlob.from_filesystem(
            datablobs_directory=self.datablobs_directory
        )
        assert_datablob_metadatas_are_equal(self.datablob, loaded)
        assert_datablobs_tfdatas_are_equal(self.datablob, loaded)

    def test_save_v3_and_load(self):
        """We we haven't implemented Save/Load version 3 yet, so we expect an exception."""
        with self.assertRaises(sp.exceptions.UnsupportedDataBlobSaveLoadVersion):
            self.datablob.save(
                datablobs_directory=self.datablobs_directory,
                save_load_version=3,
            )


class TestDataBlobSharding(unittest.TestCase):
    """
    Test that we can load and save DataBlobs with sharding enabled.
    """

    @classmethod
    def setUpClass(cls):
        cls.datablob = MyDataBlobArbitraryRows(
            hyperparams=dict(
                num_training=12,
                num_validation=13,
                num_test=3,
            ),
        )

    def _get_num_shards(self, *, datablobs_directory: str, subtype: str) -> int:
        shard_parent_directory = os.path.join(
            datablobs_directory, self.datablob.name, subtype, "tfdata"  # type: ignore
        )
        shard_dir_name = [
            name
            for name in os.listdir(shard_parent_directory)
            if name not in ("snapshot.metadata", "dataset_spec.pb")
        ][0]
        return len(os.listdir(os.path.join(shard_parent_directory, shard_dir_name)))

    def test_sharding_default(self):
        """Test that sharding works with the default parameter values."""
        expected_num_shards = 1
        with tempfile.TemporaryDirectory() as datablobs_directory:
            self.datablob.save(datablobs_directory=datablobs_directory)
            for subtype in ["training", "validation", "test"]:
                with self.subTest(subtype):
                    actual_num_shards = self._get_num_shards(
                        datablobs_directory=datablobs_directory, subtype=subtype
                    )
                    self.assertEqual(expected_num_shards, actual_num_shards)

    def test_sharding_with_shard_quantity_1(self):
        """Test that we can read/and write a DataBlob in multiple shards (with shard_quantity=1)."""
        for expected_num_shards in range(1, 4):
            with tempfile.TemporaryDirectory() as datablobs_directory:
                self.datablob.save(
                    datablobs_directory=datablobs_directory,
                    num_shards=expected_num_shards,
                )
                for subtype in ["training", "validation", "test"]:
                    with self.subTest(
                        subtype=subtype, expected_num_shards=expected_num_shards
                    ):
                        actual_num_shards = self._get_num_shards(
                            datablobs_directory=datablobs_directory, subtype=subtype
                        )
                        self.assertEqual(expected_num_shards, actual_num_shards)

                        for shard_offset in range(expected_num_shards):
                            loaded = MyDataBlobArbitraryRows.from_filesystem(
                                hyperparams=self.datablob.hyperparams,
                                datablobs_directory=datablobs_directory,
                                shard_offset=shard_offset,
                            )
                            assert_datablob_metadatas_are_equal(self.datablob, loaded)
                            expected = list(
                                getattr(self.datablob, subtype)
                                .shard(
                                    index=shard_offset, num_shards=expected_num_shards
                                )
                                .as_numpy_iterator()
                            )
                            actual = list(getattr(loaded, subtype).as_numpy_iterator())
                            self.assertEqual(expected, actual)
