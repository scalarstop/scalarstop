"""
Test the scalarstop.datablob module
"""
import itertools
import json
import os
import tempfile
import unittest

import pandas as pd
import tensorflow as tf

import scalarstop as sp
from scalarstop._filesystem import rmtree
from tests.assertions import (
    assert_datablob_dataframes_are_equal,
    assert_datablob_metadatas_are_equal,
    assert_datablobs_tfdatas_are_equal,
    assert_dataframes_are_equal,
    assert_directory,
    assert_hyperparams_are_equal,
)


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
        return tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

    def set_validation(self):
        """Set the validation tfdata."""
        return tf.data.Dataset.from_tensor_slices([6, 7, 8, 9, 10])

    def set_test(self):
        """Set the test tfdata."""
        return tf.data.Dataset.from_tensor_slices([11, 12, 13, 14, 15])


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
            dataset_directory = os.path.dirname(os.path.dirname(path))
            this_dataset_path = os.path.join(dataset_directory, self.name)
            # We have to create a directory, and then put something inside
            # the directory to make sure that we can't copy into the
            # directory without triggering an error.
            os.mkdir(this_dataset_path)
            with open(os.path.join(this_dataset_path, "training"), "w"):
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
            dataset_directory = os.path.dirname(os.path.dirname(path))
            this_dataset_path = os.path.join(dataset_directory, self.name)
            with open(os.path.join(this_dataset_path), "w"):
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


class DataBlobTestCase(unittest.TestCase):
    """Base class for unit tests involving DataBlobs."""

    def assert_saved_metadata_json(self, blob, filename):
        """Check that the metadata.json has been properly saved to the filesystem."""
        expected = dict(
            name=blob.name,
            group_name=blob.group_name,
            hyperparams=sp.dataclasses.asdict(blob.hyperparams),
        )
        with open(filename, "r") as fh:
            actual = json.load(fh)
        self.assertEqual(expected, actual)

    def assert_saved_dataframe(self, blob, subtype, this_dataset_directory):
        """Check that DataFrames have been properly saved to the filesystem."""
        current_dataframe = getattr(blob, subtype + "_dataframe")
        assert_directory(
            os.path.join(this_dataset_directory, subtype),
            ["dataframe.pickle.gz", "tfdata", "element_spec.pickle"],
        )
        # Check that the loaded dataframe is the same.
        loaded_dataframe = pd.read_pickle(
            os.path.join(this_dataset_directory, subtype, "dataframe.pickle.gz")
        )
        assert_dataframes_are_equal(current_dataframe, loaded_dataframe)

    def assertions_for_save(self, blob, dataset_directory):
        """Assert that saving a DataBlob works."""
        with self.assertRaises(FileExistsError):
            blob.save(dataset_directory)
        self.assertTrue(os.path.exists(os.path.join(dataset_directory, blob.name)))
        this_dataset_directory = os.path.join(dataset_directory, blob.name)
        assert_directory(
            this_dataset_directory,
            ["training", "validation", "test", "metadata.json", "metadata.pickle"],
        )
        self.assert_saved_metadata_json(
            blob, os.path.join(this_dataset_directory, "metadata.json")
        )
        for subtype in ["training", "validation", "test"]:
            with self.subTest(subtype):
                # If the DataBlob has dataframes, check that they have been
                # serialized too.
                current_dataframe = getattr(blob, subtype + "_dataframe", None)
                if current_dataframe is not None:
                    self.assert_saved_dataframe(blob, subtype, this_dataset_directory)
                else:
                    assert_directory(
                        os.path.join(this_dataset_directory, subtype),
                        ["tfdata", "element_spec.pickle"],
                    )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(this_dataset_directory, subtype, "tfdata")
                    )
                )

    def assertions_for_batch_cache_save(self, blob, sequence, dataset_directory):
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
        self.assertions_for_save(blob, dataset_directory)


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

    def test_no_hyperparams(self):
        """Test the error when a DataBlob has required hyperparams and we don't specify them."""
        with self.assertRaises(sp.exceptions.WrongHyperparamsKeys):
            MyDataBlobRequiredHyperparams()

    def test_unnecessary_hyperparams(self):
        """Test what happens when we pass unnecessary hyperparams to a DataBlob."""
        with self.assertRaises(sp.exceptions.WrongHyperparamsKeys):
            MyDataBlob(hyperparams=dict(z=3))

    def test_missing_hyperparams_class(self):
        """Test what happens when the hyperparams class itself is missing."""
        with self.assertRaises(sp.exceptions.YouForgotTheHyperparams):
            MyDataBlobForgotHyperparams()

    def test_save_success(self):
        """Test that we can save a :py:class:`DataBlob`."""
        with tempfile.TemporaryDirectory() as dataset_directory:
            blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            blob.save(dataset_directory)
            with self.assertRaises(FileExistsError):
                blob.save(dataset_directory)
            self.assertions_for_save(blob, dataset_directory)

            # Check that serialized element spec is correct.
            for subtype in ["training", "validation", "test"]:
                tfdata = getattr(blob, subtype)
                with open(
                    os.path.join(
                        dataset_directory, blob.name, subtype, "element_spec.pickle"
                    ),
                    "rb",
                ) as fh:
                    loaded_element_spec = sp.pickle.load(fh)
                self.assertEqual(tfdata.element_spec, loaded_element_spec)

    def test_save_catch_exception(self):
        """
        Test that :py:meth:`DataBlob.save` deletespartially-saved
        data if it fails.
        """
        with tempfile.TemporaryDirectory() as dataset_directory:
            with self.assertRaises(RuntimeError):
                DataBlobWillFailtoSave().save(dataset_directory)
            assert_directory(dataset_directory, [])

    def test_save_dataset_created_during_creation_1(self):
        """
        Test what happens when the final :py:class:`DataBlob`
        directory is created after we start (but do not finish)
        saving our :py:class:`DataBlob`.
        """
        with tempfile.TemporaryDirectory() as dataset_directory:
            with self.assertRaises(sp.exceptions.FileExistsDuringDataBlobCreation):
                DataBlobWillCauseDirectoryNotEmpty().save(dataset_directory)

    def test_save_dataset_created_during_creation_2(self):
        """
        Test what happpens when we create a file (not a directory)
        at the location that we wanted to create the directory
        to save our :py:class:`DataBlob`.
        """
        with tempfile.TemporaryDirectory() as dataset_directory:
            with self.assertRaises(sp.exceptions.FileExistsDuringDataBlobCreation):
                DataBlobWillCauseNotADirectoryError().save(dataset_directory)

    def test_load_from_directory(self):
        """Test that we can load a :py:class:`DataBlob` from the filesystem."""
        with tempfile.TemporaryDirectory() as dataset_directory:
            blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1")
            blob.save(dataset_directory)
            loaded = sp.DataBlob.load_from_directory(
                os.path.join(dataset_directory, blob.name)
            )
            assert_datablob_metadatas_are_equal(blob, loaded)
            assert_datablobs_tfdatas_are_equal(blob, loaded)

    def test_load_dataset_not_found_1(self):
        """
        Test what happens when we try to load a nonexistent
        :py:class:`DataBlob` from the filesystem.
        """
        with self.assertRaises(sp.exceptions.DataBlobNotFound):
            sp.DataBlob.load_from_directory("asdf")

    def test_load_dataset_not_found_2(self):
        """
        Test what happens when we delete a directory containing
        a :py:class:`tf.data.Dataset` and the element spec.
        """
        for deleted_subtype in ["training", "validation", "test"]:
            with tempfile.TemporaryDirectory() as dataset_directory:
                blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1").save(
                    dataset_directory
                )
                rmtree(os.path.join(dataset_directory, blob.name, deleted_subtype))
                loaded = sp.DataBlob.load_from_directory(
                    os.path.join(dataset_directory, blob.name)
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
            with tempfile.TemporaryDirectory() as dataset_directory:
                blob = MyDataBlob(hyperparams=dict(a=1, b="hi"), secret="s1").save(
                    dataset_directory
                )
                rmtree(
                    os.path.join(
                        dataset_directory, blob.name, deleted_subtype, "tfdata"
                    )
                )
                loaded = sp.DataBlob.load_from_directory(
                    os.path.join(dataset_directory, blob.name)
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
        with tempfile.TemporaryDirectory() as dataset_directory:
            operations = dict(
                cache=dict(),
                save=dict(dataset_directory=dataset_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataBlob(hyperparams=dict(a=idx, b="hi"), secret="s1")
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, dataset_directory
                    )
                    loaded = blob.load_from_directory(
                        os.path.join(dataset_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablobs_tfdatas_are_equal(blob, loaded)

    def test_batch_cache_save_load_permutations(self):
        """Test loading the dataset after batch/cache/save."""
        with tempfile.TemporaryDirectory() as dataset_directory:
            operations = dict(
                batch=dict(batch_size=2),
                cache=dict(),
                save=dict(dataset_directory=dataset_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataBlob(hyperparams=dict(a=idx, b="hi"), secret="s1")
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, dataset_directory
                    )
                    loaded = blob.load_from_directory(
                        os.path.join(dataset_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)


class Test_WrapDataBlob(DataBlobTestCase):
    """Test the :py:class:`_WrapDataBlob` class."""

    def test_not_implemented(self):
        """
        Test that :py:class:`_WrapDataBlob` methods are not
        implemented until overridden.
        """
        wrapped = sp.datablob._WrapDataBlob(wraps=MyDataBlob())
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


class Test_CacheDataBlob(DataBlobTestCase):
    """Tests for _CacheDataBlob."""

    def test_success(self):
        """Test that in-memory caching works."""
        for subtype in ["training", "validation", "test"]:
            # Each iteration increments the count by 3
            # because the tf.data pipeline is not cached.
            blob = DataBlobForCaching()
            for _ in getattr(blob, subtype):
                continue
            for _ in getattr(blob, subtype):
                continue
            for _ in getattr(blob, subtype):
                continue
            self.assertEqual(blob.count, 9)

            # Because we enable caching, the count only
            # increments the first time.
            cached = blob.cache()
            self.assertEqual(blob.name, cached.name)
            self.assertEqual(blob.group_name, cached.group_name)
            self.assertEqual(blob.hyperparams, cached.hyperparams)
            for _ in getattr(cached, subtype):
                continue
            self.assertEqual(cached.count, 12)
            for _ in getattr(cached, subtype):
                continue
            for _ in getattr(cached, subtype):
                continue
            self.assertEqual(cached.count, 12)


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
        with tempfile.TemporaryDirectory() as dataset_directory:
            blob = MyDataFrameDataBlob()
            blob.save(dataset_directory)

    def test_load_from_directory(self):
        """
        Test that we can load a :py:class:`DataFrameDataBlob`
        from the filesystem.
        """
        with tempfile.TemporaryDirectory() as dataset_directory:
            blob = MyDataFrameDataBlob()
            blob.save(dataset_directory)
            loaded = MyDataFrameDataBlob.load_from_directory(
                os.path.join(dataset_directory, blob.name)
            )
            assert_datablob_metadatas_are_equal(blob, loaded)
            assert_datablobs_tfdatas_are_equal(blob, loaded)
            assert_datablob_dataframes_are_equal(blob, loaded)

    def test_cache_save_load_permutations(self):
        """Test loading the dataset after cache and or save."""
        with tempfile.TemporaryDirectory() as dataset_directory:
            operations = dict(
                cache=dict(),
                save=dict(dataset_directory=dataset_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataFrameDataBlob(hyperparams=dict(a=idx))
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, dataset_directory
                    )
                    loaded = blob.load_from_directory(
                        os.path.join(dataset_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablobs_tfdatas_are_equal(blob, loaded)
                    assert_datablob_dataframes_are_equal(blob, loaded)

    def test_batch_cache_save_load_permutations(self):
        """Test loading the dataset after batch/cache/save."""
        with tempfile.TemporaryDirectory() as dataset_directory:
            operations = dict(
                batch=dict(batch_size=2),
                cache=dict(),
                save=dict(dataset_directory=dataset_directory),
            )
            for idx, sequence in enumerate(itertools.permutations(operations.items())):
                blob = MyDataFrameDataBlob(hyperparams=dict(a=idx))
                with self.subTest(sequence[0]):
                    self.assertions_for_batch_cache_save(
                        blob, sequence, dataset_directory
                    )
                    loaded = blob.load_from_directory(
                        os.path.join(dataset_directory, blob.name)
                    )
                    assert_datablob_metadatas_are_equal(blob, loaded)
                    assert_datablob_dataframes_are_equal(blob, loaded)
