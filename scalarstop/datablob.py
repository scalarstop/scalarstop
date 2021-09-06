"""
Group together and name your training, validation, and test sets.

The classes in this module are used to group together
data into training, validation, and test sets used for
training machine learning models. We also record the
hyperparameters used to process the dataset.

The :py:class:`DataBlob` subclass name and hyperparameters
are used to create a unique content-addressable name
that makes it easy to keep track of many datasets at once.
"""

import errno
import json
import os
from typing import Any, Mapping, Optional, Type, Union

import pandas as pd
import tensorflow as tf
from log_with_context import Logger

import scalarstop.pickle
from scalarstop._constants import (
    _DATAFRAME_FILENAME,
    _DEFAULT_SAVE_LOAD_VERSION,
    _METADATA_JSON_FILENAME,
    _METADATA_PICKLE_FILENAME,
    _SUBTYPE_TEST,
    _SUBTYPE_TRAINING,
    _SUBTYPE_VALIDATION,
    _SUBTYPES,
)
from scalarstop._filesystem import rmtree
from scalarstop._logging import Timeblock
from scalarstop._naming import temporary_filename
from scalarstop._single_namespace import SingleNamespace
from scalarstop._tfdata import tfdata_load, tfdata_save
from scalarstop.dataclasses import asdict
from scalarstop.exceptions import (
    DataBlobNotFound,
    ElementSpecNotFound,
    FileExists,
    FileExistsDuringDataBlobCreation,
    InconsistentCachingParameters,
    IsNotImplemented,
    TensorFlowDatasetNotFound,
)
from scalarstop.hyperparams import (
    AppendHyperparamsType,
    HyperparamsType,
    NestedHyperparamsType,
    enforce_dict,
)

_LOGGER = Logger(__name__)


class DataBlob(SingleNamespace):
    """
    Subclass this to group your training, validation, and test sets for training machine learning models.

    Here is how to use :py:class:`DataBlob` to group your training,
    validation, and test sets:

    1. Subclass :py:class:`DataBlob` with a class name that describes
       your dataset in general. In this example, we'll use
       ``MyDataBlob`` as the class name.

    2. Define a dataclass using the ``@sp.dataclass`` decorator at
       ``MyDataBlob.Hyperparams``. We'll define an *instance* of
       this dataclass at ``MyDataBlob.hyperparams``. This describes
       the hyperparameters involved in processing your dataset.

    3. Override the methods :py:meth:`DataBlob.set_training`,
       :py:meth:`DataBlob.set_validation`, and :py:meth:`DataBlob.set_test`
       to generate :py:class:`tf.data.Dataset` pipelines
       representing your training, validation, and test sets.

    Those three steps roughly look like:

    >>> import tensorflow as tf
    >>> import scalarstop as sp
    >>>
    >>> class MyDataBlob(sp.DataBlob):
    ...
    ...     @sp.dataclass
    ...     class Hyperparams(sp.HyperparamsType):
    ...             cols: int
    ...
    ...     def _data(self):
    ...             x = tf.random.uniform(shape=(10, self.hyperparams.cols))
    ...             y = tf.round(tf.random.uniform(shape=(10,1)))
    ...             return tf.data.Dataset.zip((
    ...                     tf.data.Dataset.from_tensor_slices(x),
    ...                     tf.data.Dataset.from_tensor_slices(y),
    ...             ))
    ...
    ...     def set_training(self):
    ...         return self._data()
    ...
    ...     def set_validation(self):
    ...         return self._data()
    ...
    ...     def set_test(self):
    ...         return self._data()
    >>>

    In our above example, our training, validation, and test sets
    are created with the exact same code. In practice, you'll
    be creating them with different inputs.

    Now we create an instance of our subclass so we can start
    using it.

    >>> datablob = MyDataBlob(hyperparams=dict(cols=3))
    >>> datablob
    <sp.DataBlob MyDataBlob-bn5hpc7ueo2uz7as1747tetn>

    :py:class:`DataBlob` instances are given a unique name
    by hashing together the class name with the instance's
    hyperparameters.

    >>> datablob.name
    'MyDataBlob-bn5hpc7ueo2uz7as1747tetn'
    >>>
    >>> datablob.group_name
    'MyDataBlob'
    >>>
    >>> datablob.hyperparams
    MyDataBlob.Hyperparams(cols=3)
    >>>
    >>> sp.enforce_dict(datablob.hyperparams)
    {'cols': 3}

    We save exactly one instance of each :py:class:`tf.data.Dataset`
    pipeline in the properties :py:attr:`DataBlob.training`,
    :py:attr:`DataBlob.validation`, and :py:attr:`DataBlob.test`.

    >>> datablob.training
    <ZipDataset shapes: ((3,), (1,)), types: (tf.float32, tf.float32)>
    >>>
    >>> datablob.validation
    <ZipDataset shapes: ((3,), (1,)), types: (tf.float32, tf.float32)>
    >>>
    >>> datablob.test
    <ZipDataset shapes: ((3,), (1,)), types: (tf.float32, tf.float32)>

    :py:class:`DataBlob` objects have some methods for applying
    :py:mod:`tf.data` transformations to the training, validation, and
    test sets at the same time:

    * **Batching.** :py:meth:`DataBlob.batch` will batch the training, validation,
      and test sets at the same time. If you call
      :py:meth:`DataBlob.batch` with the keyword argument
      ``with_tf_distribute=True``, your input batch size will be
      multiplied by the number of replicas in your :py:mod:`tf.distribute`
      strategy.

    * **Caching.** :py:meth:`DataBlob.cache` will cache the training, validation,
      and test sets in memory once you iterate over them. This is
      useful if your :py:class:`tf.data.Dataset` are doing something
      computationally expensive each time you iterate over them.

    * **Saving/loading to/from the filesystem.** :py:meth:`DataBlob.save`
      saves the training, validation, and test
      sets to a path on the filesystem. This can be loaded back with
      the classmethod :py:meth:`DataBlob.from_exact_path`.

    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.TemporaryDirectory()
    >>>
    >>> datablob = datablob.save(tempdir.name)
    >>>
    >>> os.listdir(tempdir.name)
    ['MyDataBlob-bn5hpc7ueo2uz7as1747tetn']

    >>> path = os.path.join(tempdir.name, datablob.name)
    >>> loaded_datablob = MyDataBlob.from_exact_path(path)
    >>> loaded_datablob
    <sp.DataBlob MyDataBlob-bn5hpc7ueo2uz7as1747tetn>

    Alternatively, if you have the hyperparameters of the
    :py:class:`DataBlob` but not the name, you can use the
    classmethod :py:meth:`DataBlob.from_filesystem`.

    >>> loaded_datablob_2 = MyDataBlob.from_filesystem(
    ...    hyperparams=dict(cols=3),
    ...    datablobs_directory=tempdir.name,
    ... )
    >>> loaded_datablob_2
    <sp.DataBlob MyDataBlob-bn5hpc7ueo2uz7as1747tetn>

    (and now let's clean up the temporary directory from above)

    >>> tempdir.cleanup()
    """  # pylint: disable=line-too-long

    _training: Optional[tf.data.Dataset] = None
    _validation: Optional[tf.data.Dataset] = None
    _test: Optional[tf.data.Dataset] = None

    @classmethod
    def from_filesystem(
        cls,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
    ):
        """
        Load a :py:class:`DataBlob` from the filesystem, calculating the
        filename from the hyperparameters.

        Args:
            hyperparams: The hyperparameters of the model that we want to load.

            datablobs_directory: The parent directory of all of your saved
                :py:class:`DataBlob` s. The exact filename is calculated
                from the class name and hyperparams.
        """
        name = cls.calculate_name(hyperparams=hyperparams)
        path = os.path.join(datablobs_directory, name)
        return cls.from_exact_path(
            path, shard_offset=shard_offset, shard_quantity=shard_quantity
        )

    @classmethod
    def from_filesystem_or_new(
        cls,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
        **kwargs,
    ):
        """Load a :py:class:`DataBlob` from the filesystem, calculating the
        filename from the hyperparameters. Create a new :py:class:`DataBlob`
        if we cannot find a saved one on the filesystem.

        Args:
            hyperparams: The hyperparameters of the model that we want to load.

            datablobs_directory: The parent directory of all of your saved
                :py:class:`DataBlob` s. The exact filename is calculated
                from the class name and hyperparams.

            **kwargs: Other keyword arguments that you need to pass to
                your ``__init__()``.
        """
        try:
            return cls.from_filesystem(
                hyperparams=hyperparams,
                datablobs_directory=datablobs_directory,
                shard_offset=shard_offset,
                shard_quantity=shard_quantity,
            )
        except DataBlobNotFound:
            return cls(hyperparams=hyperparams, **kwargs)

    @staticmethod
    def from_exact_path(
        path: str, *, shard_offset: Optional[int] = None, shard_quantity: int = 1
    ) -> "DataBlob":
        """Load a :py:class:`DataBlob` from a directory on the filesystem."""
        return _LoadDataBlob.from_exact_path(
            path=path, shard_offset=shard_offset, shard_quantity=shard_quantity
        )

    def exists_in_datablobs_directory(
        self,
        datablobs_directory: str,
    ) -> bool:
        """
        Returns ``True`` if this :py:class:`DataBlob` was already saved
        within ``datablobs_directory``.

        Args:
            datablobs_directory: The parent directory of all of your
                saved :py:class:`DataBlob` s.

        Returns:
            Returns ``True`` if we found a py:class:`DataBlob`
            metadata file at the expected location.
        """
        path = os.path.join(datablobs_directory, self.name)
        return os.path.exists(path)

    def __repr__(self) -> str:
        return f"<sp.DataBlob {self.name}>"

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

    def batch(
        self,
        batch_size: int,
        *,
        training: bool = True,
        validation: bool = True,
        test: bool = True,
        with_tf_distribute: bool = False,
    ) -> "DataBlob":
        """
        Batch this :py:class:`DataBlob`.

        Args:
            batch_size: The number of items to collect into a batch.

            training: Whether to batch the training set.
                Defaults to ``True``.

            validation: Whether to batch the validation set.
                Defaults to ``True``.

            test: Whether to batch the test set. Defaults to ``True``.

            with_tf_distribute: Whether to consider ``tf.distribute``
                auto-data sharding when calculating the batch size.

        """
        return _BatchDataBlob(
            wraps=self,
            batch_size=batch_size,
            training=training,
            validation=validation,
            test=test,
            with_tf_distribute=with_tf_distribute,
        )

    def cache(
        self,
        *,
        training: bool = True,
        validation: bool = True,
        test: bool = True,
        precache_training: bool = False,
        precache_validation: bool = False,
        precache_test: bool = False,
    ) -> "DataBlob":
        """
        Cache this :py:class:`DataBlob` into memory before iterating over it.

        By default, this creates a :py:class:`DataBlob` containing a
        TensorFlow ``CacheDataset`` for each of the training, validation and test
        :py:class:`tf.data.Dataset` s.

        But these datasets do not load into memory until the first time you
        *completely* iterate over one--from start to end. If you want to
        immediately load your training, validation, or test sets, you can
        set ``precache_training``, ``precache_validation``, and/or
        ``precache_test`` to ``True``.

        Args:
            training: Lazily cache the training set in CPU memory.
                Defaults to ``True``.

            validation: Lazily cache the validation set in CPU memory.
                Defaults to ``True``.

            test: Lazily cache the test set in CPU memory.
                Defaults to ``True``.

            precache_training: Eagerly cache the training set into memory.
                Defaults to ``False``.

            precache_validation: Eagerly cache the validation set into
                memory. Defaults to ``False``.

            precache_test: Eagerly cache the test set into memory.
                Defaults to ``False``.
        """
        return _CacheDataBlob(
            wraps=self,
            training=training,
            validation=validation,
            test=test,
            precache_training=precache_training,
            precache_validation=precache_validation,
            precache_test=precache_test,
        )

    def with_options(
        self,
        options: tf.data.Options,
        *,
        training: bool = True,
        validation: bool = True,
        test: bool = True,
    ) -> "DataBlob":
        """
        Apply a :py:class:`tf.data.Options` object to this :py:class:`DataBlob`.

        Args:
            options: The :py:class:`tf.data.Options` object to apply.

            training: Apply the options to the training set. Defaults to ``True``.

            validation: Apply the options to the validation set. Defaults to ``True``.

            test: Apply the options to the test set. Defaults to ``True``.
        """
        return _WithOptionsDataBlob(
            wraps=self,
            options=options,
            training=training,
            validation=validation,
            test=test,
        )

    def save_hook(  # pylint: disable=unused-argument
        self, *, subtype: str, path: str
    ) -> None:
        """
        Override this method to run additional code when saving this
        :py:class:`DataBlob` to disk.
        """
        return None

    def save(
        self,
        datablobs_directory: str,
        *,
        ignore_existing: bool = False,
        num_shards: int = 1,
        save_load_version: int = _DEFAULT_SAVE_LOAD_VERSION,
    ) -> "DataBlob":
        """
        Save this :py:class:`DataBlob` to disk.

        Args:
            datablobs_directory: The directory where you plan on storing all of your
                DataBlobs. This method will save this :py:class:`DataBlob` in a subdirectory
                of ``datablobs_directory`` with same name as :py:attr:`DataBlob.name`.

            ignore_existing: Set this to ``True`` to ignore if
                there is already a :py:class:`DataBlob` at the given
                path.

            save_load_version: The ScalarStop version for the ScalarStop protocol.

        Returns:
            Return ``self``, enabling you to place this call in a chain.
        """
        # Begin writing our hyperparameters, dataframes, tfdata, and element spec.
        final_datablob_path = os.path.join(datablobs_directory, self.name)
        if os.path.exists(final_datablob_path):
            if ignore_existing:
                return self
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

            # These are the fields that we are going to identically save
            # in both the JSON and the Pickle.
            common_metadata = dict(
                name=self.name,
                group_name=self.group_name,
                save_load_version=save_load_version,
                num_shards=num_shards,
            )

            with open(metadata_json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    obj=dict(
                        **common_metadata,
                        # When we save hyperparams to JSON, we first.
                        # convert Python dataclasses to dictionaries.
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
                        **common_metadata,
                        # When we save hyperparams to Pickle, we make
                        # sure to serialize the hyperparams as a Python
                        # dataclass.
                        hyperparams=self.hyperparams,
                    ),
                    file=fh,
                )

            for subtype in _SUBTYPES:
                # Create the directory for each subtype.
                subtype_path = os.path.join(temp_datablob_path, subtype)
                tfdata_dataset = getattr(self, subtype)
                tfdata_save(
                    dataset=tfdata_dataset,
                    path=subtype_path,
                    num_shards=num_shards,
                    save_load_version=save_load_version,
                )

                # Save additional elements that subclasses want to save.
                self.save_hook(subtype=subtype, path=subtype_path)

        except BaseException:
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
    """
    Subclass this to transform a :py:class:`pandas.DataFrame` into your training, validation, and test sets.

    :py:class:`DataBlob` is useful when you want to manually define your
    :py:mod:`tf.data` pipelines and their input tensors.

    However, if your input tensors are in a fixed-size list or
    :py:class:`~pandas.DataFrame` that you want to *slice* into
    a training, validation, and test set, then you might find
    :py:class:`DataFrameDataBlob` handy.

    Here is how to use it:

    1. Subclass :py:class:`DataFrameDataBlob` with a class name that
       describes your dataset.

    2. Override :py:meth:`DataFrameDataBlob.set_dataframe` and have
       it return a *single* :py:class:`~pandas.DataFrame` that contains
       all of the *inputs* for your training, validation, and
       test sets. The :py:class:`~pandas.DataFrame` should have
       one column representing training samples and another
       column representing training labels.

    3. Override :py:meth:`DataFrameDataBlob.transform` and define
       a method that transforms an arbitrary :py:class:`~pandas.DataFrame`
       of *inputs* into a :py:class:`tf.data.Dataset` pipeline
       that represents the actual dataset needed for training
       and evaluation.

    We define what fraction of the :py:class:`~pandas.DataFrame` to split
    with the class attributes :py:attr:`DataFrameDataBlob.training_fraction`
    and :py:attr:`DataFrameDataBlob.validation_fraction`. By default,
    60 percent of the :py:class:`~pandas.DataFrame` is marked
    for the training set, 20 percent for the validation set,
    and the remainder of the :py:class:`~pandas.DataFrame` for the test set.

    Roughly, this looks like:

    >>> import pandas as pd
    >>> import tensorflow as tf
    >>> import scalarstop as sp
    >>>
    >>> class MyDataFrameDataBlob(sp.DataFrameDataBlob):
    ...    samples_column: str = "samples"
    ...    labels_column: str = "labels"
    ...    training_fraction: float = 0.6
    ...    validation_fraction: float = 0.2
    ...
    ...    @sp.dataclass
    ...    class Hyperparams(sp.HyperparamsType):
    ...        length: int = 0
    ...
    ...    def set_dataframe(self):
    ...        samples = list(range(self.hyperparams.length))
    ...        labels = list(range(self.hyperparams.length))
    ...        return pd.DataFrame({self.samples_column: samples, self.labels_column: labels})
    ...
    ...    def transform(self, dataframe: pd.DataFrame):
    ...        return tf.data.Dataset.zip((
    ...                tf.data.Dataset.from_tensor_slices(dataframe[self.samples_column]),
    ...                tf.data.Dataset.from_tensor_slices(dataframe[self.labels_column]),
    ...        ))

    >>> datablob2 = MyDataFrameDataBlob(hyperparams=dict(length=10))

    And you can use the resulting object in all of the same ways as
    we've demonstrated with :py:class:`DataBlob` subclass instances above.
    """  # pylint: disable=line-too-long

    samples_column: str = "samples"
    labels_column: str = "labels"
    training_fraction: float = 0.6
    validation_fraction: float = 0.2

    _dataframe: Optional[pd.DataFrame] = None
    _training_dataframe: Optional[pd.DataFrame] = None
    _validation_dataframe: Optional[pd.DataFrame] = None
    _test_dataframe: Optional[pd.DataFrame] = None

    @staticmethod
    def from_exact_path(
        path: str,
        *,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
    ) -> Union[DataBlob, "DataFrameDataBlob"]:
        """Load a :py:class:`DataFrameDataBlob` from a directory on the filesystem."""
        loaded = _LoadDataFrameDataBlob.from_exact_path(
            path=path,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
        )
        return loaded

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

    def save_hook(self, *, subtype: str, path: str) -> None:
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
    def __init__(
        self, wraps: DataBlob, *, training: bool, validation: bool, test: bool
    ):  # pylint: disable=super-init-not-called
        self._wraps = wraps
        self.Hyperparams = self._wraps.Hyperparams
        self.hyperparams = self._wraps.hyperparams
        self._name = wraps.name
        self._group_name = wraps.group_name
        self._enable_training = training
        self._enable_validation = validation
        self._enable_test = test

    def __getattribute__(self, key: str):
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

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        raise IsNotImplemented("_WrapDataBlob._wrap_tfdata()")

    def set_training(self) -> tf.data.Dataset:
        if self._enable_training:
            return self._wrap_tfdata(self._wraps.set_training())
        return self._wraps.set_training()

    @property
    def training(self) -> tf.data.Dataset:
        """An instance of the training set tf.data."""
        if self._enable_training:
            if self._training is None:
                self._training = self._wrap_tfdata(self._wraps.training)
            return self._training
        return self._wraps.training

    def set_validation(self) -> tf.data.Dataset:
        if self._enable_validation:
            return self._wrap_tfdata(self._wraps.set_validation())
        return self._wraps.set_validation()

    @property
    def validation(self) -> tf.data.Dataset:
        """An instance of the validation set tf.data."""
        if self._enable_validation:
            if self._validation is None:
                self._validation = self._wrap_tfdata(self._wraps.validation)
            return self._validation
        return self._wraps.validation

    def set_test(self) -> tf.data.Dataset:
        if self._enable_test:
            return self._wrap_tfdata(self._wraps.set_test())
        return self._wraps.set_test()

    @property
    def test(self) -> tf.data.Dataset:
        """An instance of the test set tf.data."""
        if self._enable_test:
            if self._test is None:
                self._test = self._wrap_tfdata(self._wraps.test)
            return self._test
        return self._wraps.test

    def save_hook(self, *, subtype: str, path: str) -> None:
        return self._wraps.save_hook(subtype=subtype, path=path)


class AppendDataBlob(DataBlob):
    """
    Subclass this to create a new :py:class:`DataBlob` that extends an existing :py:class:`DataBlob`.

    The :py:class:`AppendDataBlob` class is useful when you have an existing
    :py:class:`DataBlob` or :py:class:`DataFrameDataBlob` with most, but not *all*
    of the functionality you need. If you are trying to implement multiple
    data pipelines that share a common compute-intensive first step, you
    can implement your pipelines as :py:class:`AppendDataBlob` subclasses with
    the common first step as a :py:class:`DataBlob` that you save and load
    to/from the filesystem.

    Let's begin by creating a :py:class:`DataBlob` that we will use as a
    parent for an :py:class:`AppendDataBlob`.

    >>> import tensorflow as tf
    >>> import scalarstop as sp
    >>>
    >>> class MyDataBlob(sp.DataBlob):
    ...
    ...     @sp.dataclass
    ...     class Hyperparams(sp.HyperparamsType):
    ...             length: int
    ...
    ...     def _data(self):
    ...         length = self.hyperparams.length
    ...         x = tf.data.Dataset.from_tensor_slices(list(range(0, length)))
    ...         y = tf.data.Dataset.from_tensor_slices(list(range(length, length * 2)))
    ...         return tf.data.Dataset.zip((x, y))
    ...
    ...     def set_training(self):
    ...         return self._data()
    ...
    ...     def set_validation(self):
    ...         return self._data()
    ...
    ...     def set_test(self):
    ...         return self._data()
    >>>

    And then we create an instance of the datablob and save it to
    the filesystem.

    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.TemporaryDirectory()
    >>>
    >>> datablob = MyDataBlob(hyperparams=dict(length=5))
    >>> datablob
    <sp.DataBlob MyDataBlob-dac936v7mb1ue9phjp6tc3sb>
    >>>
    >>> list(datablob.training.as_numpy_iterator())
    [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    >>>
    >>> datablob = datablob.save(tempdir.name)
    >>>
    >>> os.listdir(tempdir.name)
    ['MyDataBlob-dac936v7mb1ue9phjp6tc3sb']

    Now, let's say that we want to create an :py:class:`AppendDataBlob`
    that takes in any input :py:class:`DataBlob` or
    :py:class:`DataFrameDataBlob` and multiplies every number in every tensor by
    a constant.

    >>> class MyAppendDataBlob(sp.AppendDataBlob):
    ...
    ...     @sp.dataclass
    ...     class Hyperparams(sp.AppendHyperparamsType):
    ...          coefficient: int
    ...
    ...     hyperparams: "MyAppendDataBlob.Hyperparams"
    ...
    ...     def __init__(self, *, parent: sp.DataBlob, hyperparams):
    ...         hyperparams_dict = sp.enforce_dict(hyperparams)
    ...         if hyperparams_dict["coefficient"] < 1:
    ...             raise ValueError("Coefficient is too low.")
    ...         super().__init__(parent=parent, hyperparams=hyperparams_dict)
    ...
    ...     def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
    ...          return tfdata.map(
    ...              lambda x, y: (
    ...                  x * self.hyperparams.coefficient,
    ...                  y * self.hyperparams.coefficient,
    ...               )
    ...          )
    >>>
    >>> append = MyAppendDataBlob(parent=datablob, hyperparams=dict(coefficient=3))
    >>> list(append.training.as_numpy_iterator())
    [(0, 15), (3, 18), (6, 21), (9, 24), (12, 27)]

    (And now let's clean up the temporary directory that we created earlier.)

    >>> tempdir.cleanup()
    """  # pylint: disable=line-too-long

    Hyperparams: Type[AppendHyperparamsType] = AppendHyperparamsType

    @classmethod
    def create_append_hyperparams(
        cls,
        *,
        parent: DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
    ):
        """
        Combine the hyperparams from the parent :py:class:`DataBlob` with
        the hyperparams meant for this :py:class:`AppendDataBlob`.
        """
        return dict(
            parent=NestedHyperparamsType(
                name=parent.name,
                group_name=parent.group_name,
                hyperparams=parent.hyperparams,
            ),
            **enforce_dict(hyperparams),
        )

    @classmethod
    def calculate_name_from_parent(
        cls,
        *,
        parent: DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
    ):
        """Calculate the hashed name of this :py:class:`AppendDataBlob`,
        given the hyperparameters and the parent :py:class:`DataBlob`."""
        append_hyperparams = cls.create_append_hyperparams(
            parent=parent, hyperparams=hyperparams
        )
        return cls.calculate_name(hyperparams=append_hyperparams)

    @classmethod
    def from_filesystem_with_parent(
        cls,
        *,
        parent: DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
    ):
        """
        Load a :py:class:`AppendDataBlob` from the filesystem, calculating the
        filename from the parent and the hyperparameters..
        """
        name = cls.calculate_name_from_parent(parent=parent, hyperparams=hyperparams)
        path = os.path.join(datablobs_directory, name)
        return cls.from_exact_path(
            path, shard_offset=shard_offset, shard_quantity=shard_quantity
        )

    @classmethod
    def from_filesystem_or_new_with_parent(  # pylint: disable=arguments-differ
        cls,
        *,
        parent: DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]],
        datablobs_directory: str,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
        **kwargs,
    ):
        """
        Load a :py:class:`AppendDataBlob` from the filesystem, calculating the
        filename from the hyperparameters. Create a new :py:class:`AppendDataBlob`
        if we cannot find a saved one on the filesystem.

        Args:
            parent: The parent :py:class:`DataBlob` to extend.

            hyperparams: The hyperparameters of the  :py:class:`DataBlob`
                that we want to load.

            datablobs_directory: The parent directory of all of your saved
                :py:class:`DataBlob` s. The exact filename is calculated.
        """
        try:
            return cls.from_filesystem_with_parent(
                parent=parent,
                hyperparams=hyperparams,
                datablobs_directory=datablobs_directory,
                shard_offset=shard_offset,
                shard_quantity=shard_quantity,
            )
        except DataBlobNotFound:
            return cls(parent=parent, hyperparams=hyperparams, **kwargs)

    def __init__(
        self,
        *,
        parent: DataBlob,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        **kwargs,
    ):
        """
        Args:
            parent: The :py:class:`DataBlob` to extend.

            hyperparams: Additional hyperparameters to add on top of the
                existing hyperparameters from the parent :py:class:`DataBlob`.
        """
        super().__init__(
            hyperparams=self.create_append_hyperparams(
                parent=parent, hyperparams=hyperparams
            )
        )
        self._parent = parent

    def __getattribute__(self, key: str):
        """
        Returns attributes on the wrapped object for any key that isn't present on this object.

        Args:
            key: The key to look up on this object (and then on the wrapped object).
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            parent = object.__getattribute__(self, "_parent")
            return parent.__getattribute__(key)

    @property
    def parent(self) -> DataBlob:
        """The parent :py:class:`DataBlob`."""
        return self._parent

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        """
        Override this to add something to the training, validation,
        and test :py:class:`tf.data.Dataset` pipelines.
        """
        raise IsNotImplemented("AppendDataBlob._wrap_tfdata()")

    def set_training(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._parent.set_training())

    @property
    def training(self) -> tf.data.Dataset:
        if self._training is None:
            self._training = self._wrap_tfdata(self._parent.training)
        return self._training

    def set_validation(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._parent.set_validation())

    @property
    def validation(self) -> tf.data.Dataset:
        if self._validation is None:
            self._validation = self._wrap_tfdata(self._parent.validation)
        return self._validation

    def set_test(self) -> tf.data.Dataset:
        return self._wrap_tfdata(self._parent.set_test())

    @property
    def test(self) -> tf.data.Dataset:
        if self._test is None:
            self._test = self._wrap_tfdata(self._parent.test)
        return self._test


class _BatchDataBlob(_WrapDataBlob):
    """
    Batch the training, validation, and test sets.

    Optionally, you can batch the dataset in accordance with the
    currently-active :py:class:`tf.distribute.Strategy` strategy.
    Tensorflow Distribute Strategies split batches up betweeen
    each replica device, so the common convention is to first multiply
    the batch size by the number of available replicas.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        batch_size: int,
        training: bool,
        validation: bool,
        test: bool,
        with_tf_distribute: bool = False,
    ):
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._input_batch_size = batch_size
        self._with_tf_distribute = with_tf_distribute
        if self._with_tf_distribute:
            self._final_batch_size: int = (
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


def _precache_tfdata(tfdata: tf.data.Dataset) -> tf.data.Dataset:
    """Tensorflow CacheDatasets will internally cache themselves if you iterate
    over them."""
    for _ in tfdata:
        continue
    return tfdata


class _CacheDataBlob(_WrapDataBlob):
    """
    Cache this :py:class:`DataBlob` in memory.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        training: bool,
        validation: bool,
        test: bool,
        precache_training: bool,
        precache_validation: bool,
        precache_test: bool,
    ):
        if training is False and precache_training is True:
            raise InconsistentCachingParameters(
                "You cannot pass `training=False` and `precache_training=True` "
                "to `DataBlob.cache()`. If you want `precache_training=True`, "
                "then set `training=True`, which it is by default."
            )
        if validation is False and precache_validation is True:
            raise InconsistentCachingParameters(
                "You cannot pass `validation=False` and `precache_validation=True` "
                "to `DataBlob.cache()`. If you want `precache_validation=True`, "
                "then set `validation=True`, which it is by default."
            )
        if test is False and precache_test is True:
            raise InconsistentCachingParameters(
                "You cannot pass `test=False` and `precache_test=True` "
                "to `DataBlob.cache()`. If you want `precache_test=True`, "
                "then set `test=True`, which it is by default."
            )
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._precache_training = precache_training
        self._precache_validation = precache_validation
        self._precache_test = precache_test

        if self._precache_training:
            with Timeblock(
                name=f"(pre)caching {self.name}/training", print_function=_LOGGER.info
            ):
                self._training = _precache_tfdata(
                    self._wrap_tfdata(self._wraps.training)
                )

        if self._precache_validation:
            with Timeblock(
                name=f"(pre)caching {self.name}/validation", print_function=_LOGGER.info
            ):
                self._validation = _precache_tfdata(
                    self._wrap_tfdata(self._wraps.validation)
                )

        if self._precache_test:
            with Timeblock(
                name=f"(pre)caching {self.name}/validation", print_function=_LOGGER.info
            ):
                self._test = _precache_tfdata(self._wrap_tfdata(self._wraps.test))

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.cache()


class _WithOptionsDataBlob(_WrapDataBlob):
    """
    Apply a :py:class:`tf.data.Options` object to this :py:class:`DataBlob`.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        options: tf.data.Options,
        training: bool,
        validation: bool,
        test: bool,
    ):
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._tfdata_options = options

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        with_options_tfdata = tfdata.with_options(self._tfdata_options)
        return with_options_tfdata


class _LoadDataBlob(DataBlob):
    """Loads a saved :py:class:`DataBlob` from the filesystem."""

    def __init__(
        self,
        *,
        path: str,
        name: str,
        group_name: str,
        hyperparams: HyperparamsType,
        total_num_shards: int,
        save_load_version: int,
        shard_offset: Optional[int],
        shard_quantity: int,
    ):
        self.Hyperparams = hyperparams.__class__
        super().__init__(hyperparams=hyperparams)
        self._path = path
        self._name = name
        self._group_name = group_name
        self._save_load_version = save_load_version
        self._total_num_shards = total_num_shards
        self._shard_offset = shard_offset
        self._shard_quantity = shard_quantity

    @classmethod
    def from_exact_path(
        cls,
        path: str,
        *,
        shard_offset: Optional[int] = None,
        shard_quantity: int = 1,
    ) -> Union[DataBlob, DataFrameDataBlob]:
        metadata_path = os.path.join(path, _METADATA_PICKLE_FILENAME)
        try:
            with open(metadata_path, "rb") as fh:
                metadata = scalarstop.pickle.load(fh)
        except FileNotFoundError as exc:
            raise DataBlobNotFound(path) from exc
        name = metadata["name"]
        group_name = metadata["group_name"]
        save_load_version = metadata.get("save_load_version", 1)
        total_num_shards = metadata.get("num_shards", 1)
        hyperparams = metadata["hyperparams"]
        return cls(
            path=path,
            name=name,
            group_name=group_name,
            hyperparams=hyperparams,
            total_num_shards=total_num_shards,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
            save_load_version=save_load_version,
        )

    def _load_tfdata(self, subtype: str) -> tf.data.Dataset:
        """Load one of the :py:class:`tf.data.Dataset` s that we have saved."""
        try:
            return tfdata_load(
                path=os.path.join(self._path, subtype),
                total_num_shards=self._total_num_shards,
                shard_offset=self._shard_offset,
                shard_quantity=self._shard_quantity,
                save_load_version=self._save_load_version,
            )
        except ElementSpecNotFound as exc:
            raise TensorFlowDatasetNotFound(self._path) from exc

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
        return pd.read_pickle(os.path.join(self._path, subtype, _DATAFRAME_FILENAME))

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
