"""A class that generates a training, validation, and test set from a set of hyperparameters."""  # pylint: disable=line-too-long

import errno
import json
import os
from typing import Any, Mapping, Optional, Union

import pandas as pd
import tensorflow as tf

import scalarstop.pickle
from scalarstop._filesystem import rmtree
from scalarstop._naming import temporary_filename
from scalarstop.dataclasses import asdict
from scalarstop.exceptions import (
    DataBlobNotFound,
    ElementSpecNotFound,
    FileExists,
    FileExistsDuringDataBlobCreation,
    IsNotImplemented,
    TensorFlowDatasetNotFound,
)
from scalarstop.hyperparams import HyperparamsType, hash_hyperparams, init_hyperparams

_METADATA_JSON_FILENAME = "metadata.json"
_METADATA_PICKLE_FILENAME = "metadata.pickle"
_ELEMENT_SPEC_FILENAME = "element_spec.pickle"
_DATAFRAME_FILENAME = "dataframe.pickle.gz"
_SUBTYPE_TRAINING = "training"
_SUBTYPE_VALIDATION = "validation"
_SUBTYPE_TEST = "test"
_TFDATA_DIRECTORY_NAME = "tfdata"
_SUBTYPES = (_SUBTYPE_TRAINING, _SUBTYPE_VALIDATION, _SUBTYPE_TEST)


def _load_tfdata_dataset(path, element_spec=None):
    """
    Load a :py:class:`tf.data.Dataset` from a filesystem path.

    This is a little different from
    :py:func:`tf.data.experimental.load` because we save the
    `element_spec` in a pickled file above the
    :py:class:`tf.data.Dataset` 's directory.

    If you want to read a dataset that doesn't have the
    ``element_spec`` saved on disk, then just specify
    the ``element_spec`` keyword argument with your own value.
    """
    # Load the element spec.
    if element_spec is None:
        element_spec_path = os.path.join(path, _ELEMENT_SPEC_FILENAME)
        try:
            with open(element_spec_path, "rb") as fp:
                element_spec = scalarstop.pickle.load(file=fp)
        except FileNotFoundError as exc:
            raise ElementSpecNotFound(path) from exc

    # Load the tf.data Dataset.
    tfdata_path = os.path.join(path, _TFDATA_DIRECTORY_NAME)
    try:
        return tf.data.experimental.load(
            path=tfdata_path,
            element_spec=element_spec,
        )
    except tf.errors.NotFoundError as exc:
        raise TensorFlowDatasetNotFound(tfdata_path) from exc


class DataBlob:
    """Subclass this and override :py:meth:`~DataBlob.set_training`, :py:meth:`~DataBlob.set_validation`, and :py:meth:`~DataBlob.set_test` to create :py:class:`tf.data.Dataset` pipelines."""  # pylint: disable=line-too-long

    Hyperparams = HyperparamsType
    hyperparams: HyperparamsType

    _training: Optional[tf.data.Dataset] = None
    _validation: Optional[tf.data.Dataset] = None
    _test: Optional[tf.data.Dataset] = None

    _name: Optional[str] = None
    _group_name: Optional[str] = None

    def __init__(
        self, *, hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None
    ):
        self.hyperparams = init_hyperparams(
            self=self,
            hyperparams=hyperparams,
            hyperparams_class=self.Hyperparams,
        )

    @staticmethod
    def load_from_directory(this_dataset_directory) -> "DataBlob":
        """Load a :py:class:`DataBlob` from a directory on the filesystem."""
        return _LoadDataBlob.load_from_directory(
            this_dataset_directory=this_dataset_directory
        )

    def __repr__(self) -> str:
        return f"<sp.DataBlob {self.name}>"

    @property
    def name(self) -> str:
        """
        The name of this specific dataset.

        If you intend on overriding this method, make sure that
        instances of the same class with different hyperparameters
        will have different names.

        However, if you have additional parameters to your
        subclass's ``__init__()`` or configuration that are not
        hyperparameters to your :py:class:`DataBlob`, then they
        should not change the name. For example, you might want to
        pass a database connection string or a filesystem path
        to your :py:class:`DataBlob` subclass. Other users of your
        subclass might have different values for those parameters,
        but they don't change the actual dataset itself--and
        therefore should not change the :py:class:`DataBlob` name.
        """
        if self._name is None:
            self._name = "-".join((self.group_name, hash_hyperparams(self.hyperparams)))
        return self._name

    @property
    def group_name(self) -> str:
        """
        The group name of this dataset.

        This is typically the :py:class:`DataBlob` subclass's class name.

        Conceptually, the group name is the name for all :py:class:`DataBlob` s that
        share the same code but have different hyperparameters.
        """
        if self._group_name is None:
            self._group_name = self.__class__.__name__
        return self._group_name

    def set_training(self) -> tf.data.Dataset:
        """Create a :py:class:`tf.data.Dataset` for the training set."""
        raise IsNotImplemented("DataBlob.set_training()")

    @property
    def training(self) -> tf.data.Dataset:
        """A :py:class:`tf.data.Dataset` instance representing the training set."""
        if self._training is None:
            self._training = self.set_training()
        return self._training

    def set_validation(self) -> tf.data.Dataset:
        """Create a :py:class:`tf.data.Dataset` for the validation set."""
        raise IsNotImplemented("DataBlob.set_validation()")

    @property
    def validation(self) -> tf.data.Dataset:
        """A :py:class:`tf.data.Dataset` instance representing the validation set."""
        if self._validation is None:
            self._validation = self.set_validation()
        return self._validation

    def set_test(self) -> tf.data.Dataset:
        """Create a :py:class:`tf.data.Dataset` for the test set."""
        raise IsNotImplemented("DataBlob.set_test()")

    @property
    def test(self) -> tf.data.Dataset:
        """A :py:class:`tf.data.Dataset` instance representing the test set."""
        if self._test is None:
            self._test = self.set_test()
        return self._test

    def batch(self, batch_size: int, *, with_tf_distribute: bool = False) -> "DataBlob":
        """Batch this :py:class:`DataBlob`."""
        return _BatchDataBlob(
            wraps=self, batch_size=batch_size, with_tf_distribute=with_tf_distribute
        )

    def cache(self) -> "DataBlob":
        """Cache this :py:class:`DataBlob` into memory before iterating over it."""
        return _CacheDataBlob(wraps=self)

    def save_hook(self, *, subtype, path) -> None:  # pylint: disable=unused-argument
        """
        Override this method to run additional code when saving this
        :py:class:`DataBlob` to disk.
        """
        return None

    def save(self, dataset_directory) -> "DataBlob":
        """
        Save this :py:class:`DataBlob` to disk.

        Args:
            dataset_directory: The directory where you plan on storing all of your
                DataBlobs. This method will save this :py:class:`DataBlob` in a subdirectory
                of ``dataset_directory`` with same name as :py:attr:`DataBlob.name`.

        Returns:
            Return ``self``, enabling you to place this call in a chain.
        """
        # Begin writing our hyperparameters, dataframes, tfdata, and element spec.
        final_datablob_path = os.path.join(dataset_directory, self.name)
        if os.path.exists(final_datablob_path):
            raise FileExists(
                f"File or directory already exists at path {final_datablob_path}"
            )
        temp_datablob_path = temporary_filename(final_datablob_path)
        os.mkdir(temp_datablob_path)
        try:
            # Save metadata as JSON so it is human-readable on the filesystem.
            metadata_json_path = os.path.join(
                temp_datablob_path, _METADATA_JSON_FILENAME
            )
            with open(metadata_json_path, "w") as fh:
                json.dump(
                    obj=dict(
                        name=self.name,
                        group_name=self.group_name,
                        hyperparams=asdict(self.hyperparams),
                    ),
                    fp=fh,
                    sort_keys=True,
                    indent=4,
                )
            # Save the hyperparameter again in the pickle format.
            # This is better for actually deserializing them.
            metadata_pickle_path = os.path.join(
                temp_datablob_path, _METADATA_PICKLE_FILENAME
            )
            with open(metadata_pickle_path, "wb") as fh:  # type: ignore
                scalarstop.pickle.dump(
                    obj=dict(
                        name=self.name,
                        group_name=self.group_name,
                        hyperparams=self.hyperparams,
                    ),
                    file=fh,
                )

            for subtype in _SUBTYPES:
                # Create the directory for each subtype.
                subtype_path = os.path.join(temp_datablob_path, subtype)
                os.mkdir(subtype_path)

                # Save the tf.data Dataset.
                tfdata_path = os.path.join(subtype_path, _TFDATA_DIRECTORY_NAME)
                tfdata_dataset = getattr(self, subtype)
                tf.data.experimental.save(
                    dataset=tfdata_dataset,
                    path=tfdata_path,
                )

                # Save the element spec.
                element_spec_path = os.path.join(subtype_path, _ELEMENT_SPEC_FILENAME)
                with open(element_spec_path, "wb") as fh:  # type: ignore
                    scalarstop.pickle.dump(
                        obj=tfdata_dataset.element_spec,
                        file=fh,
                    )

                # Save additional elements that subclasses want to save.
                self.save_hook(subtype=subtype, path=subtype_path)

        except BaseException as exc:
            # If we run into an error, delete our partially constructed dataset
            # from the filesystem.
            rmtree(temp_datablob_path)
            raise
        else:
            # Now that we have completed the construction of our dataset,
            # we are going to move it from the temporary directory name
            # to the permanent directory name.
            file_exists_exception = FileExistsDuringDataBlobCreation(
                "Failed to rename dataset directory from "
                f"{temp_datablob_path} to {final_datablob_path}. "
                f"The directory at {final_datablob_path} became occupied "
                "during the dataset creation."
            )
            try:
                os.rename(temp_datablob_path, final_datablob_path)
            except (FileExistsError, NotADirectoryError) as exc:
                raise file_exists_exception from exc
            except OSError as exc:
                if exc.errno == errno.ENOTEMPTY:
                    raise file_exists_exception from exc
                raise exc
            else:
                return self


class DataFrameDataBlob(DataBlob):
    """Subclass this to create a a :py:class:`DataBlob` where the resulting :py:class:`tf.data.Dataset` pipelines are each fed a :py:class:`pandas.DataFrame` of input tensors."""  # pylint: disable=line-too-long

    samples_column: str = "samples"
    labels_column: str = "labels"
    training_fraction: float = 0.6
    validation_fraction: float = 0.2

    _dataframe: Optional[pd.DataFrame] = None
    _training_dataframe: Optional[pd.DataFrame] = None
    _validation_dataframe: Optional[pd.DataFrame] = None
    _test_dataframe: Optional[pd.DataFrame] = None

    @staticmethod
    def load_from_directory(this_dataset_directory) -> "DataFrameDataBlob":
        """Load a :py:class:`DataFrameDataBlob` from a directory on the filesystem."""
        return _LoadDataFrameDataBlob.load_from_directory(
            this_dataset_directory=this_dataset_directory
        )

    def __repr__(self) -> str:
        return f"<sp.DataFrameDataBlob {self.name}>"

    def set_dataframe(self) -> pd.DataFrame:
        """
        Create a new :py:class:`pandas.DataFrame` that contains all of the data for
        the training, validation, and test sets.
        """
        raise IsNotImplemented("DataFrameDataBlob.set_dataframe()")

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        A :py:class:`pandas.DataFrame` that represents the entire
        training, validation, and test set.
        """
        if self._dataframe is None:
            self._dataframe = self.set_dataframe()
        return self._dataframe

    def set_training_dataframe(self) -> pd.DataFrame:
        """
        Sets the :py:class:`pandas.DataFrame` for the training set.

        By default, this method slices the :py:class:`pandas.DataFrame` you have supplied to
        :py:meth:`set_dataframe`.

        Alternatively, you can choose to directly subclass
        :py:meth:`set_training_dataframe`, :py:meth:`set_validation_dataframe`,
        and :py:meth`set_test_dataframe`.

        Returns:
            Returns a :py:class:`pandas.DataFrame`.
        """
        end = int(len(self.dataframe) * self.training_fraction)
        return self.dataframe[:end]

    @property
    def training_dataframe(self) -> pd.DataFrame:
        """A :py:class:`pandas.DataFrame` representing training set input tensors."""
        if self._training_dataframe is None:
            self._training_dataframe = self.set_training_dataframe()
        return self._training_dataframe

    def set_validation_dataframe(self) -> pd.DataFrame:
        """
        Sets the :py:class:`pandas.DataFrame` for the validation set.

        By default, this method slices the :py:class:`pandas.DataFrame` you have supplied to
        :py:meth:`set_dataframe`.

        Alternatively, you can choose to directly subclass
        :py:meth:`set_training_dataframe`, :py:meth:`set_validation_dataframe`,
        and :py:meth`set_test_dataframe`.

        Returns:
            Returns a :py:class:`pandas.DataFrame`.
        """
        start = int(len(self.dataframe) * self.training_fraction)
        end = int(
            len(self.dataframe) * (self.training_fraction + self.validation_fraction)
        )
        return self.dataframe[start:end]

    @property
    def validation_dataframe(self) -> pd.DataFrame:
        """A :py:class:`pandas.DataFrame` representing validation set input tensors."""
        if self._validation_dataframe is None:
            self._validation_dataframe = self.set_validation_dataframe()
        return self._validation_dataframe

    def set_test_dataframe(self) -> pd.DataFrame:
        """
        Sets the :py:class:`pandas.DataFrame` for the test set.

        By default, this method slices the DataFrame you have supplied to
        :py:meth:`set_dataframe`.

        Alternatively, you can choose to directly subclass
        :py:meth:`set_training_dataframe`, :py:meth:`set_validation_dataframe`,
        and :py:meth`set_test_dataframe`.

        Returns:
            Returns a Pandas :py:class:`pandas.DataFrame`.
        """
        start = int(
            len(self.dataframe) * (self.training_fraction + self.validation_fraction)
        )
        return self.dataframe[start:]

    @property
    def test_dataframe(self) -> pd.DataFrame:
        """A :py:class:`pandas.DataFrame` representing test set input tensors."""
        if self._test_dataframe is None:
            self._test_dataframe = self.set_test_dataframe()
        return self._test_dataframe

    def transform(self, dataframe: pd.DataFrame) -> tf.data.Dataset:
        """Transforms any input tensors into an output :py:class:`tf.data.Dataset`."""
        raise IsNotImplemented("DataFrameDataBlob.transform()")

    def set_training(self) -> tf.data.Dataset:
        return self.transform(self.training_dataframe)

    def set_validation(self) -> tf.data.Dataset:
        return self.transform(self.validation_dataframe)

    def set_test(self) -> tf.data.Dataset:
        return self.transform(self.test_dataframe)

    def save_hook(self, *, subtype, path) -> None:
        super().save_hook(subtype=subtype, path=path)
        dataframe = getattr(self, subtype + "_dataframe", None)
        dataframe_path = os.path.join(path, _DATAFRAME_FILENAME)
        dataframe.to_pickle(
            path=dataframe_path,
            compression="gzip",
            protocol=scalarstop.pickle._PICKLE_PROTOCOL_VERSION,
        )


class _WrapDataBlob(DataBlob):
    """Wraps the training, validation, and test sets using a single function."""

    # We do not call super().__init__() because that call would be affected
    # by our custom __getattribute__(). Thankfully, our __getattribute__()
    # takes care of any setup that super().__init__() would handle.
    def __init__(self, wraps: DataBlob):  # pylint: disable=super-init-not-called
        self._wraps = wraps
        self.Hyperparams = self._wraps.Hyperparams
        self.hyperparams = self._wraps.hyperparams
        self._name = wraps.name
        self._group_name = wraps.group_name

    def __getattribute__(self, key):
        """
        Returns attributes on the wrapped object for any key that isn't present on this object.

        Args:
            key: The key to look up on this object (and then on the wrapped object).
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            wraps = object.__getattribute__(self, "_wraps")
            return wraps.__getattribute__(key)

    def _wrap_tfdata(self, tfdata) -> tf.data.Dataset:
        raise IsNotImplemented("_WrapDataBlob._wrap_tfdata()")

    def set_training(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._wraps.set_training())

    @property
    def training(self) -> tf.data.Dataset:
        """An instance of the training set tf.data."""
        if self._training is None:
            self._training = self._wrap_tfdata(self._wraps.training)
        return self._training

    def set_validation(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._wraps.set_validation())

    @property
    def validation(self) -> tf.data.Dataset:
        """An instance of the validation set tf.data."""
        if self._validation is None:
            self._validation = self._wrap_tfdata(self._wraps.validation)
        return self._validation

    def set_test(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._wraps.set_test())

    @property
    def test(self) -> tf.data.Dataset:
        """An instance of the test set tf.data."""
        if self._test is None:
            self._test = self._wrap_tfdata(self._wraps.test)
        return self._test

    def save_hook(self, *, subtype, path) -> None:
        return self._wraps.save_hook(subtype=subtype, path=path)


class _BatchDataBlob(_WrapDataBlob):
    """
    Batch the training, validation, and test sets.

    Optionally, you can batch the dataset in accordance with the
    currently-active :py:class:`tf.distribute.Strategy` strategy.
    Tensorflow Distribute Strategies split batches up betweeen
    each replica device, so the common convention is to first multiply
    the batch size by the number of available replicas.
    """

    def __init__(self, *, wraps, batch_size: int, with_tf_distribute: bool = False):
        super().__init__(wraps=wraps)
        self._input_batch_size = batch_size
        self._with_tf_distribute = with_tf_distribute
        if self._with_tf_distribute:
            self._final_batch_size = (
                self._input_batch_size
                * tf.distribute.get_strategy().num_replicas_in_sync
            )
        else:
            self._final_batch_size = self._input_batch_size

    @property
    def batch_size(self) -> int:
        """The final batch size--taking the TensorFlow strategy into account if needed."""
        return self._final_batch_size

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.batch(self._final_batch_size)


class _CacheDataBlob(_WrapDataBlob):
    """
    Cache this :py:class:`DataBlob` in memory.
    """

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.cache()


class _LoadDataBlob(DataBlob):
    """Loads a saved :py:class:`DataBlob` from the filesystem."""

    def __init__(
        self,
        *,
        this_dataset_directory: str,
        name: str,
        group_name: str,
        hyperparams: HyperparamsType,
    ):
        self.Hyperparams = hyperparams.__class__
        super().__init__(hyperparams=hyperparams)
        self._this_dataset_directory = this_dataset_directory
        self._name = name
        self._group_name = group_name

    @classmethod
    def load_from_directory(cls, this_dataset_directory):
        metadata_path = os.path.join(this_dataset_directory, _METADATA_PICKLE_FILENAME)
        try:
            with open(metadata_path, "rb") as fh:
                metadata = scalarstop.pickle.load(fh)
        except FileNotFoundError as exc:
            raise DataBlobNotFound(this_dataset_directory) from exc
        name = metadata["name"]
        group_name = metadata["group_name"]
        hyperparams = metadata["hyperparams"]
        return cls(
            this_dataset_directory=this_dataset_directory,
            name=name,
            group_name=group_name,
            hyperparams=hyperparams,
        )

    def _load_tfdata(self, subtype) -> tf.data.Dataset:
        """Load one of the :py:class:`tf.data.Dataset` s that we have saved."""
        try:
            return _load_tfdata_dataset(
                os.path.join(self._this_dataset_directory, subtype)
            )
        except ElementSpecNotFound as exc:
            raise TensorFlowDatasetNotFound(self._this_dataset_directory) from exc

    def set_training(self) -> tf.data.Dataset:
        """An instance of the training set tf.data."""
        return self._load_tfdata(_SUBTYPE_TRAINING)

    def set_validation(self) -> tf.data.Dataset:
        """Create a :py:class:`tf.data.Dataset` for the validation set."""
        return self._load_tfdata(_SUBTYPE_VALIDATION)

    def set_test(self) -> tf.data.Dataset:
        """Create a :py:class:`tf.data.Dataset` for the test set."""
        return self._load_tfdata(_SUBTYPE_TEST)


class _LoadDataFrameDataBlob(_LoadDataBlob, DataFrameDataBlob):
    """Loads a saved :py:class:`DataFrameDataBlob` from the filesystem."""

    def _set_subtype_dataframe(self, subtype: str) -> pd.DataFrame:
        return pd.read_pickle(
            os.path.join(self._this_dataset_directory, subtype, _DATAFRAME_FILENAME)
        )

    def set_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            (
                self.training_dataframe,
                self.validation_dataframe,
                self.test_dataframe,
            )
        )

    def set_training_dataframe(self) -> pd.DataFrame:
        return self._set_subtype_dataframe(_SUBTYPE_TRAINING)

    def set_validation_dataframe(self) -> pd.DataFrame:
        return self._set_subtype_dataframe(_SUBTYPE_VALIDATION)

    def set_test_dataframe(self) -> pd.DataFrame:
        return self._set_subtype_dataframe(_SUBTYPE_TEST)
