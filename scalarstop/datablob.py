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
import os
from typing import Any, Mapping, Optional, Type, Union

import pandas as pd
import tensorflow as tf
from log_with_context import Logger

import scalarstop.pickle
from scalarstop._constants import (
    _DATAFRAME_FILENAME,
    _DEFAULT_SAVE_LOAD_VERSION,
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
from scalarstop.datablob_metadata import DataBlobMetadata
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


class DataBlobBase(SingleNamespace):
    """The abstract base class describing the properties common to all DataBlobs."""

    def __init__(
        self,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        **kwargs,
    ):
        super().__init__(hyperparams=hyperparams, **kwargs)
        self._training: Any = None
        self._validation: Any = None
        self._test: Any = None

    def set_training(self) -> Any:
        """Creates and returns a new object representing the training set."""
        raise IsNotImplemented("DataBlobBase.set_training()")

    @property
    def training(self) -> Any:
        """An object representing the training set."""
        if self._training is None:
            self._training = self.set_training()
        return self._training

    def set_validation(self) -> Any:
        """Creates and returns a new object representing the validation set."""
        raise IsNotImplemented("DataBlobBase.set_validation()")

    @property
    def validation(self) -> Any:
        """An object representing the validation set."""
        if self._validation is None:
            self._validation = self.set_validation()
        return self._validation

    def set_test(self) -> Any:
        """Creates and returns a new object representing the test set."""
        raise IsNotImplemented("DataBlobBase.set_test()")

    @property
    def test(self) -> Any:
        """An object representing the test set."""
        if self._test is None:
            self._test = self.set_test()
        return self._test


class DataBlob(DataBlobBase):
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
        Loads a :py:class:`DataBlob` from the filesystem, calculating the
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
    def from_filesystem_distributed(
        cls,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.Strategy] = None,
    ) -> "DistributedDataBlob":
        """
        Loads a sharded :py:class:`DataBlob` from the filesystem,
        automatically splitting the shards amongs the input workers
        of a :py:class:`tf.distribute.Strategy`.

        Args:
            hyperparams: The hyperparameters of the model that we want to load.

            datablobs_directory: The parent directory of all of your saved
                :py:class:`DataBlob` s. The exact filename is calculated
                from the class name and hyperparams.

            repeat: Repeats the :py:class:`DataBlob` after loading it.
                Set to ``True`` to enable infinite repeating.
                Set to a positive integer ``n`` to repeat the
                :py:class:`DataBlob` ``n`` times.
                Set to ``False`` to disable repeating.

            per_replica_batch_size: The batch size for each individual
                :py:mod:`tf.distribute` replica. This is the global
                batch size divided by :py:attr:`tf.distribute.Strategy.num_replicas_in_sync`.

            tf_distribute_strategy: The :py:class:`tf.distribute.Strategy`
                subclass to use. Optionally, this method will detect if it
                is already inside a `:py:meth:`tf.distribute.Strategy.scope`
                context manager.
        """
        return _DistributedDataBlobFromFilesystem(
            hyperparams=hyperparams,
            datablobs_directory=datablobs_directory,
            datablob_class=cls,
            repeat=repeat,
            per_replica_batch_size=per_replica_batch_size,
            tf_distribute_strategy=tf_distribute_strategy,
        )

    @classmethod
    def metadata_from_filesystem(
        cls,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
    ) -> DataBlobMetadata:
        """
        Loads this :py:class:`DataBlob` 's :py:class:`DataBlobMetadata`
        from the filesystem, calculating the filename from the hyperparameters.

        Args:
            hyperparams: The hyperparameters of the model that we want to load.

            datablobs_directory: The parent directory of all of your saved
                :py:class:`DataBlob` s. The exact filename is calculated
                from the class name and hyperparams.
        """
        name = cls.calculate_name(hyperparams=hyperparams)
        path = os.path.join(datablobs_directory, name)
        return cls.metadata_from_exact_path(path)

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

    @classmethod
    def from_exact_path_distributed(
        cls,
        *,
        path: str,
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.get_strategy] = None,
    ) -> "DistributedDataBlob":
        """
        Args:
            path: The exact location of the saved :py:class:`DataBlob`
                on the filesystem.

            repeat: Repeats the :py:class:`DataBlob` after loading it.
                Set to ``True`` to enable infinite repeating.
                Set to a positive integer ``n`` to repeat the
                :py:class:`DataBlob` ``n`` times.
                Set to ``False`` to disable repeating.

            per_replica_batch_size: The batch size for each individual
                :py:mod:`tf.distribute` replica. This is the global
                batch size divided by :py:attr:`tf.distribute.Strategy.num_replicas_in_sync`.

            tf_distribute_strategy: The :py:class:`tf.distribute.Strategy`
                subclass to use. Optionally, this method will detect if it
                is already inside a `:py:meth:`tf.distribute.Strategy.scope`
                context manager.
        """
        return _DistributedDataBlobFromExactPath(
            path=path,
            datablob_class=cls,
            repeat=repeat,
            per_replica_batch_size=per_replica_batch_size,
            tf_distribute_strategy=tf_distribute_strategy,
        )

    @staticmethod
    def metadata_from_exact_path(path: str) -> DataBlobMetadata:
        """
        Loads this :py:class:`DataBlob` 's :py:class:`DataBlobMetadata`
        from a directory on the filesystem.
        """
        return DataBlobMetadata.load(path)

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

    def prefetch(
        self,
        buffer_size: int,
        *,
        training: bool = True,
        validation: bool = True,
        test: bool = True,
    ) -> "DataBlob":
        """
        Creates a :py:class:`DataBlob` that prefetches elements for
        performance.

        Args:
            buffer_size: The maximum number of elements that will
                be buffered when prefetching. If the value
                :py:meth:`tf.data.experimental.AUTOTUNE` is used,
                then the buffer is dynamically tuned.

            training: Apply the repeat operator to the training set.
                Defaults to ``True``.

            validation: Apply the repeat operator to the validation set.
                Defaults to ``True``.

            test: Apply the repeat operator to the test set.
                Defaults to ``True``.
        """
        return _PrefetchDataBlob(
            wraps=self,
            buffer_size=buffer_size,
            training=training,
            validation=validation,
            test=test,
        )

    def repeat(
        self,
        count: Optional[int] = None,
        *,
        training: bool = True,
        validation: bool = True,
        test: bool = True,
    ) -> "DataBlob":
        """
        Repeats this :py:class:`DataBlob`.

        Args:
            count: Represents the number of times that the
                elements in the :py:class:`tf.data.Dataset` should
                be repeated. The default behavior (if ``count`` is
                ``None`` or ``-1``) is for the dataset be repeated
                indefinitely.

            training: Apply the repeat operator to the training set.
                Defaults to ``True``.

            validation: Apply the repeat operator to the validation set.
                Defaults to ``True``.

            test: Apply the repeat operator to the test set.
                Defaults to ``True``.
        """
        return _RepeatDataBlob(
            wraps=self,
            count=count,
            training=training,
            validation=validation,
            test=test,
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
            # Save DataBlob metadata to JSON and Pickle.
            DataBlobMetadata.from_datablob(
                datablob=self,
                save_load_version=save_load_version,
                num_shards=num_shards,
            ).save(temp_datablob_path)

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
        self._hyperparams = self._wraps.hyperparams
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


class _PrefetchDataBlob(_WrapDataBlob):
    """
    Prefetch elements from this :py:class:`DataBlob` for improved latency.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        buffer_size: int,
        training: bool,
        validation: bool,
        test: bool,
    ):
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._buffer_size = buffer_size

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.prefetch(self._buffer_size)


class _RepeatDataBlob(_WrapDataBlob):
    """
    Repeats this :py:class:`DataBlob` a given number of times.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        count: Optional[int] = None,
        training: bool,
        validation: bool,
        test: bool,
    ):
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._count = count

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.repeat(self._count)


class _ShardDataBlob(_WrapDataBlob):
    """
    Shards this :py:class:`DataBlob`.
    """

    def __init__(
        self,
        *,
        wraps: Any,
        num_shards: int,
        shard_index: int,
        training: bool,
        validation: bool,
        test: bool,
    ):
        super().__init__(
            wraps=wraps, training=training, validation=validation, test=test
        )
        self._num_shards = num_shards
        self._shard_index = shard_index

    def _wrap_tfdata(self, tfdata: tf.data.Dataset) -> tf.data.Dataset:
        return tfdata.shard(self._num_shards, self._shard_index)


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
        metadata = DataBlobMetadata.load(path)
        return cls(
            path=path,
            name=metadata.name,
            group_name=metadata.group_name,
            hyperparams=metadata.hyperparams,
            total_num_shards=metadata.num_shards,
            save_load_version=metadata.save_load_version,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
        )

    @property
    def total_num_shards(self) -> int:
        """
        The total number of shards for this :py:class:`DataBlob` that were
        saved to the filesystem.
        """
        return self._total_num_shards

    @property
    def shard_offset(self) -> Optional[int]:
        """
        The start offset for the range of shards that this :py:class:`DataBlob`
        is loading from the filesystem.
        """
        return self._shard_offset

    @property
    def shard_quantity(self) -> int:
        """
        The number of shards that this :py:class:`DataBlob` is loading
        from the filesystem.
        """
        return self._shard_quantity

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


class DistributedDataBlob(DataBlobBase):
    """
    Wraps a :py:class:`DataBlob` to create a TensorFlow :py:class:`tf.distribute.DistributedDataset`.

    A :py:class:`DataBlob` contains three TensorFlow :py:class:`tf.data.Dataset`
    pipelines, representing a training, validation, and test set.
    The :py:class:`DistributedDataBlob`  wraps the creation of a :py:class:`DataBlob`
    to turn each :py:class:`tf.data.Dataset` into a :py:class:`tf.distribute.DistributedDataset`
    which is used to distribute a dataset across multiple workers according
    to a :py:class:`tf.distribute.Strategy`.

    If you have saved a :py:class:`DataBlob` to the filesystem with
    :py:meth:`DataBlob.save`, then you can automatically load the :py:class:`DataBlob`
    from the filesystem as a :py:class:`DistributedDataBlob` using the
    classmethod :py:meth:`DataBlob.from_filesystem_distributed` or
    :py:meth:`DataBlob.from_exact_path_distributed`.

    For more fine-grained control, you can subclass :py:class:`DistributedDataBlob`
    and override :py:meth:`DistributedDataBlob.new_sharded_datablob` with
    your own :py:class:`DataBlob` creation and sharding logic. Optionally,
    you can also subclass :py:meth:`DistributedDataBlob.transform_datablob`
    to change how :py:class:`DistributedDataBlob` handles repeating and batching.
    Finally, you can also subclass :py:meth:`DistributedDataBlob.postprocess_tfdata`
    to make changes to individual :py:class:`tf.data.Dataset` instances rather
    than the :py:class:`DataBlob` as a whole.
    """  # pylint: disable=line-too-long

    def __init__(
        self,
        *,
        name: str,
        group_name: str,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        hyperparams_class: Type[HyperparamsType],
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.get_strategy] = None,
    ):
        """
        Args:
            name: The name of the wrapped :py:class:`DataBlob`.

            group_name: The group name of the wrapped :py:class:`DataBlob`.

            hyperparams: The hyperparameters of the wrapped :py:class:`DataBlob`.

            hyperparams_class: The :py:class:`HyperparamsType` class that
                ``hyperparams`` instances are created from.

            repeat: Repeats the :py:class:`DataBlob` after loading it.
                Set to ``True`` to enable infinite repeating.
                Set to a positive integer ``n`` to repeat the
                :py:class:`DataBlob` ``n`` times.
                Set to ``False`` to disable repeating.

            per_replica_batch_size: The batch size for each individual
                :py:mod:`tf.distribute` replica. This is the global
                batch size divided by :py:attr:`tf.distribute.Strategy.num_replicas_in_sync`.

            tf_distribute_strategy: The :py:class:`tf.distribute.Strategy`
                subclass to use. Optionally, this method will detect if it
                is already inside a `:py:meth:`tf.distribute.Strategy.scope`
                context manager.
        """
        self._name = name
        self._group_name = group_name
        self.Hyperparams = hyperparams_class
        super().__init__(
            hyperparams=hyperparams,
        )
        self._repeat = repeat
        self._per_replica_batch_size = per_replica_batch_size
        self._tf_distribute_strategy = (
            tf_distribute_strategy or tf.distribute.get_strategy()
        )
        self._training: Optional[tf.data.Dataset] = None
        self._validation: Optional[tf.data.Dataset] = None
        self._test: Optional[tf.data.Dataset] = None

    def __repr__(self) -> str:
        return f"<sp.DistributedDataBlob {self.name}>"

    def new_sharded_datablob(
        self, ctx: tf.distribute.InputContext  # pylint: disable=unused-argument
    ) -> DataBlob:
        """
        Subclass this method to return a sharded :py:class:`DataBlob`.

        Args:
            ctx: A :py:class:`tf.distribute.InputContext` instance.
                The attribute :py:attr:`tf.distribute.InputContext.input_pipeline_id`
                returns the current input pipeline.
                The attribute :py:attr:`tf.distribute.InputContext.num_input_pipelines`
                returns the total number of distributed input pipelines
                in the current :py:class:`tf.distribute.Strategy`.

        """
        raise NotImplementedError(
            "Subclass DistributedDataBlob.new_sharded_datablob() to return a sharded DataBlob."
        )

    def transform_datablob(  # pylint: disable=unused-argument
        self,
        datablob: DataBlob,
        ctx: tf.distribute.InputContext,
    ) -> DataBlob:
        """
        Transforms an already-initialized :py:class:`DataBlob` to add repeating and sharding logic.

        Args:
            datablob: The already-initialized :py:class:`DataBlob`.

            ctx: A :py:class:`tf.distribute.InputContext` instance.
                The attribute :py:attr:`tf.distribute.InputContext.input_pipeline_id`
                returns the current input pipeline.
                The attribute :py:attr:`tf.distribute.InputContext.num_input_pipelines`
                returns the total number of distributed input pipelines
                in the current :py:class:`tf.distribute.Strategy`.

        Returns:
            Returns a :py:class:`DataBlob` that has been modified by
                repeating, batching, or another transformation.
        """
        if self._repeat is True:
            datablob = datablob.repeat()
        elif self._repeat is not False:
            datablob = datablob.repeat(count=self._repeat)
        if self._per_replica_batch_size is not None:
            datablob = datablob.batch(
                batch_size=self._per_replica_batch_size,
            )
        return datablob

    def postprocess_tfdata(  # pylint: disable=unused-argument
        self,
        tfdata: tf.data.Dataset,
        ctx: tf.distribute.InputContext,
    ) -> tf.data.Dataset:
        """
        Performs additional :py:class:`tf.data.Dataset` transformations
        before turning them into :py:class:`tf.distribute.DistributedDataset`
        instances.

        Currently, the implementation in :py:class:`DistributedDataBlob`
        does nothing, but is avaiable for you to subclass and change.

        Args:
            tfdata: The input :py:class:`tf.data.Dataset` instance to transform.

            ctx: A :py:class:`tf.distribute.InputContext` instance.
                The attribute :py:attr:`tf.distribute.InputContext.input_pipeline_id`
                returns the current input pipeline.
                The attribute :py:attr:`tf.distribute.InputContext.num_input_pipelines`
                returns the total number of distributed input pipelines
                in the current :py:class:`tf.distribute.Strategy`.

        Returns:
            Returns a transformed :py:class:`tf.data.Dataset`.
        """
        return tfdata

    def _get_distributed_tfdata(self, subtype: str) -> tf.distribute.DistributedDataset:
        """
        Returns a :py:class:`tf.distribute.DistributedDataset` for the
        training, validation, or test set.

        Args:
            subtype: Either ``"training"``, ``"validation"``, or ``"test``.
        """

        def _inner(ctx: tf.distribute.InputContext):
            datablob = self.new_sharded_datablob(ctx)
            datablob = self.transform_datablob(datablob, ctx)
            if subtype == _SUBTYPE_TRAINING:
                tfdata = datablob.set_training()
            elif subtype == _SUBTYPE_VALIDATION:
                tfdata = datablob.set_validation()
            elif subtype == _SUBTYPE_TEST:
                tfdata = datablob.set_test()
            else:
                raise RuntimeError(f"Invalid {subtype=}.")
            return self.postprocess_tfdata(ctx=ctx, tfdata=tfdata)

        return self._tf_distribute_strategy.distribute_datasets_from_function(_inner)

    @property
    def tf_distribute_strategy(self) -> tf.distribute.Strategy:
        """Returns the currently-active :py:class:`tf.distribute.Strategy`."""
        return self._tf_distribute_strategy

    def set_training(self) -> tf.distribute.DistributedDataset:
        """Creates a new :py:class:`tf.distribute.DistributedDataset` for the training set."""
        return self._get_distributed_tfdata(_SUBTYPE_TRAINING)

    @property
    def training(self) -> tf.distribute.DistributedDataset:
        """A :py:class:`tf.distribute.DistributedDataset` instance for the training set."""
        if self._training is None:
            self._training = self.set_training()
        return self._training

    def set_validation(self) -> tf.distribute.DistributedDataset:
        """Creates a new :py:class:`tf.distribute.DistributedDataset` for the validation set."""
        return self._get_distributed_tfdata(_SUBTYPE_VALIDATION)

    @property
    def validation(self) -> tf.distribute.DistributedDataset:
        """A :py:class:`tf.distribute.DistributedDataset` instance for the validation set."""
        if self._validation is None:
            self._validation = self.set_validation()
        return self._validation

    def set_test(self) -> tf.distribute.DistributedDataset:
        """Creates a new :py:class:`tf.distribute.DistributedDataset` for the test set."""
        return self._get_distributed_tfdata(_SUBTYPE_TEST)

    @property
    def test(self) -> tf.distribute.DistributedDataset:
        """A :py:class:`tf.distribute.DistributedDataset` instance for the test set."""
        if self._test is None:
            self._test = self.set_test()
        return self._test


class _DistributedDataBlobFromFilesystem(DistributedDataBlob):
    """Implements :py:meth:`DataBlob.from_filesystem_distributed`."""

    def __init__(
        self,
        *,
        hyperparams: Optional[Union[Mapping[str, Any], HyperparamsType]] = None,
        datablobs_directory: str,
        datablob_class: Optional[Type[DataBlob]] = None,
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.get_strategy] = None,
    ):
        if datablob_class is None:
            self._datablob_class = DataBlob
        else:
            self._datablob_class = datablob_class
        metadata = self._datablob_class.metadata_from_filesystem(
            hyperparams=hyperparams,
            datablobs_directory=datablobs_directory,
        )
        super().__init__(
            name=metadata.name,
            group_name=metadata.group_name,
            hyperparams=metadata.hyperparams,
            hyperparams_class=metadata.hyperparams.__class__,
            repeat=repeat,
            per_replica_batch_size=per_replica_batch_size,
            tf_distribute_strategy=tf_distribute_strategy,
        )
        self._datablobs_directory = datablobs_directory
        self._num_shards = metadata.num_shards

    def new_sharded_datablob(self, ctx: tf.distribute.InputContext):
        shard_quantity = self._num_shards // ctx.num_input_pipelines
        shard_offset = shard_quantity * ctx.input_pipeline_id
        return self._datablob_class.from_filesystem(
            hyperparams=self._hyperparams,
            datablobs_directory=self._datablobs_directory,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
        )


class _DistributedDataBlobFromExactPath(DistributedDataBlob):
    """Implements :py:meth:`DataBlob.from_exact_path_distributed`."""

    def __init__(
        self,
        *,
        path: str,
        datablob_class: Optional[Type[DataBlob]] = None,
        repeat: Union[bool, int, None] = True,
        per_replica_batch_size: Optional[int] = None,
        tf_distribute_strategy: Optional[tf.distribute.get_strategy] = None,
    ):
        if datablob_class is None:
            self._datablob_class = DataBlob
        else:
            self._datablob_class = datablob_class
        metadata = self._datablob_class.metadata_from_exact_path(
            path=path,
        )
        super().__init__(
            name=metadata.name,
            group_name=metadata.group_name,
            hyperparams=metadata.hyperparams,
            hyperparams_class=metadata.hyperparams.__class__,
            repeat=repeat,
            per_replica_batch_size=per_replica_batch_size,
            tf_distribute_strategy=tf_distribute_strategy,
        )
        self._path = path
        self._num_shards = metadata.num_shards

    def new_sharded_datablob(self, ctx: tf.distribute.InputContext):
        shard_quantity = self._num_shards // ctx.num_input_pipelines
        shard_offset = shard_quantity * ctx.input_pipeline_id
        return DataBlob.from_exact_path(
            path=self._path,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
        )
