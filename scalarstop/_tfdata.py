"""Internal code for saving and loading :py:class:`tf.data.Dataset` pipelines."""
import os
from typing import Callable, Optional

import tensorflow as tf

import scalarstop.pickle
from scalarstop._constants import _ELEMENT_SPEC_FILENAME, _TFDATA_DIRECTORY_NAME
from scalarstop._cpu import num_usable_virtual_cpu_cores
from scalarstop.exceptions import (
    DataBlobShardingNotSupported,
    DataBlobShardingValueError,
    ElementSpecNotFound,
    TensorFlowDatasetNotFound,
    UnsupportedDataBlobSaveLoadVersion,
)
from scalarstop.warnings import warn_deprecated

_ENUMERATE_IDX_ELEMENT_SPEC = tf.TensorSpec(shape=(), dtype=tf.int64, name=None)


def _undo_enumerate(_idx, row):
    """
    A :py:meth:`tf.data.Dataset.map` function for
    reversing the :py:meth:`tf.data.Dataset.enumerate()`
    transformation.

    This function takes a :py:class:`tf.data.Dataset`
    of shape ``(idx, row)`` and returns the ``row``.
    """
    return row


def make_num_shards_on_save(num_shards: int) -> Callable:
    """
    Generates a sharding function for :py:func`tf.data.experimental.save`.

    Args:
        num_shards: The number of distinct files to save to the filesystem.

    Returns:
        Returns a function that accepts an *enumerated*
        :py:class:`tf.data.Dataset` and returns the enumerated
        index modulo `num_shards`.
    """

    def shard(idx, _row):
        return idx % num_shards

    return shard


def _load_v1(
    *,
    tfdata_path: str,
    element_spec,
    shard_offset: Optional[int],
    shard_quantity: int,
    total_num_shards: int,
) -> tf.data.Dataset:
    """
    tfdata loader v1.

    This version does not support sharding. To shard your
    :py:class:`DataBlob`, use version 3 or newer.
    """
    if shard_offset is not None or total_num_shards != 1:
        raise DataBlobShardingNotSupported(
            version=1,
            offset=shard_offset,
            quantity=shard_quantity,
            total_num_shards=total_num_shards,
        )
    return tf.data.experimental.load(
        path=tfdata_path,
        element_spec=element_spec,
    )


def _select_these_shards_v2(
    offset: Optional[int], quantity: int, total_num_shards: int
):
    """
    Generates a sharding function for :py:func:`tf.data.experimental.load`.

    This function does not set the correct ``cycle_length`` when
    interleaving datasets. Please use :py:func:`select_these_shards_v3`
    or a higher version.

    Args:
        offset: The first shard index to select from the filesystem. If
            this value is ``None``, then we will load all shards.

        quantity: The number of consecutive shards to load
            starting from (and including) the shard located
            at ``offset``. This argument has no effect
            if ``offset`` is ``None``.

        total_num_shards: The total number of shards in the
            saved :py:class:`tf.data.Dataset`.

    Returns:
        Returns a function that returns an individual
        shard as a :py:class:`tf.data.Dataset`.
    """

    if total_num_shards < 1:
        raise DataBlobShardingValueError(
            f"`{total_num_shards=}` cannot be less than 1."
        )

    if offset is not None:
        if offset >= total_num_shards:
            raise DataBlobShardingValueError(
                f"{offset=} is a shard index that cannot be >= {total_num_shards=}."
            )

        if quantity > total_num_shards:
            raise DataBlobShardingValueError(
                f"{quantity=} cannot be greater than > {total_num_shards=}."
            )

        offset_quantity_sum = offset + quantity
        if offset_quantity_sum > total_num_shards:
            raise DataBlobShardingValueError(
                f"The sum of {offset=} and {quantity=} ({offset_quantity_sum}) "
                f"cannot be greater than {total_num_shards=}."
            )

    def select(datasets):
        if offset is not None:
            retval = datasets.skip(offset).take(quantity)
        else:
            retval = datasets
        return retval.interleave(lambda x: x)

    return select


def _load_v2(
    *,
    tfdata_path: str,
    element_spec,
    shard_offset: Optional[int],
    shard_quantity: int,
    total_num_shards: int,
) -> tf.data.Dataset:
    """
    tfdata loader v2.

    This version is DEPRECATED because it does not return
    elements in order when reading from multiple shards at once.
    Onlu use this version if you have existing code and
    trained models that depend on this function's broken
    behavior.
    """
    warn_deprecated(
        "You are loading a DataBlob with ScalarStop Load/Save version 2. "
        "This version is DEPRECATED because it does not load elements "
        "from the saved DataBlob when attempting to load multiple "
        "shards at once. Please recreate your saved DataBlobs to"
        "migrate to Load/Save version 3."
    )
    # The v2 `save()` function calls `enumerate()` on the dataset
    # before saving. This changes the `element_spec`, and we have
    # to account for it here.
    element_spec_after_enumerate = (_ENUMERATE_IDX_ELEMENT_SPEC, element_spec)
    return tf.data.experimental.load(
        path=tfdata_path,
        element_spec=element_spec_after_enumerate,
        reader_func=_select_these_shards_v2(
            offset=shard_offset,
            quantity=shard_quantity,
            total_num_shards=total_num_shards,
        ),
    ).map(_undo_enumerate)


def _select_these_shards_v3(
    offset: Optional[int], quantity: int, total_num_shards: int
):
    """
    Generates a sharding function for :py:func:`tf.data.experimental.load`.

    Args:
        offset: The first shard index to select from the filesystem. If
            this value is ``None``, then we will load all shards.

        quantity: The number of consecutive shards to load
            starting from (and including) the shard located
            at ``offset``. This argument has no effect
            if ``offset`` is ``None``.

        total_num_shards: The total number of shards in the
            saved :py:class:`tf.data.Dataset`.

    Returns:
        Returns a function that returns a :py:class:`tf.data.Dataset`
        that selects individual shards.
    """

    if total_num_shards < 1:
        raise DataBlobShardingValueError(
            f"`{total_num_shards=}` cannot be less than 1."
        )

    if offset is not None:
        if offset >= total_num_shards:
            raise DataBlobShardingValueError(
                f"{offset=} is a shard index that cannot be >= {total_num_shards=}."
            )

        if quantity > total_num_shards:
            raise DataBlobShardingValueError(
                f"{quantity=} cannot be greater than > {total_num_shards=}."
            )

        offset_quantity_sum = offset + quantity
        if offset_quantity_sum > total_num_shards:
            raise DataBlobShardingValueError(
                f"The sum of {offset=} and {quantity=} ({offset_quantity_sum}) "
                f"cannot be greater than {total_num_shards=}."
            )

        cycle_length = quantity
    else:
        cycle_length = total_num_shards

    # In our tests, we found that setting num_parallel_calls to
    # tf.data.experimental.AUTOTUNE makes loading datasets twice as slow.
    # Instead, we'll set the number of parallel calls to the number
    # of hyperthreaded CPU cores available to the current process on
    # this machine--unless the number of shards that we are loading
    # is less than the number of CPU cores.
    num_cpus = num_usable_virtual_cpu_cores()
    if num_cpus:
        num_parallel_calls = min(cycle_length, num_cpus)
    else:
        # If we were unable to probe the number of virtual
        # CPU cores on the current machine, then we'll let
        # TensorFlow deal with the problem.
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    def select(datasets):
        """The actual TensorFlow function for selecting datasets."""
        if offset is not None:
            retval = datasets.skip(offset).take(quantity)
        else:
            retval = datasets
        return retval.interleave(
            lambda x: x,
            cycle_length=cycle_length,
            num_parallel_calls=num_parallel_calls,
            deterministic=True,
        )

    return select


def _load_v3(
    *,
    tfdata_path: str,
    element_spec,
    shard_offset: Optional[int],
    shard_quantity: int,
    total_num_shards: int,
) -> tf.data.Dataset:
    """
    tfdata loader v3.

    This fixes an issue with tfdata loader v2 where
    rows from shards were being returned in the incorrect
    order because we didn't fix the ``cycle_length``
    when interleaving datasets.
    """
    # The v2 `save()` function calls `enumerate()` on the dataset
    # before saving. This changes the `element_spec`, and we have
    # to account for it here.
    element_spec_after_enumerate = (_ENUMERATE_IDX_ELEMENT_SPEC, element_spec)
    return tf.data.experimental.load(
        path=tfdata_path,
        element_spec=element_spec_after_enumerate,
        reader_func=_select_these_shards_v3(
            offset=shard_offset,
            quantity=shard_quantity,
            total_num_shards=total_num_shards,
        ),
    ).map(
        _undo_enumerate,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    )


def _load(
    *,
    tfdata_path: str,
    element_spec,
    shard_offset: Optional[int],
    shard_quantity: int,
    save_load_version: int,
    total_num_shards: int,
):
    """load a :py:class:`tf.data.Dataset`."""
    if save_load_version == 1:
        return _load_v1(
            tfdata_path=tfdata_path,
            element_spec=element_spec,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
            total_num_shards=total_num_shards,
        )
    if save_load_version == 2:
        return _load_v2(
            tfdata_path=tfdata_path,
            element_spec=element_spec,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
            total_num_shards=total_num_shards,
        )
    if save_load_version == 3:
        return _load_v3(
            tfdata_path=tfdata_path,
            element_spec=element_spec,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
            total_num_shards=total_num_shards,
        )
    raise UnsupportedDataBlobSaveLoadVersion(
        version=save_load_version,
    )


def tfdata_load(
    *,
    path: str,
    save_load_version,
    total_num_shards: int = 1,
    element_spec=None,
    shard_offset: Optional[int] = None,
    shard_quantity: int = 1,
) -> tf.data.Dataset:
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
        loaded_tfdata = _load(
            tfdata_path=tfdata_path,
            element_spec=element_spec,
            shard_offset=shard_offset,
            shard_quantity=shard_quantity,
            save_load_version=save_load_version,
            total_num_shards=total_num_shards,
        )
    except tf.errors.NotFoundError as exc:
        raise TensorFlowDatasetNotFound(tfdata_path) from exc

    # Tell TensorFlow that we want it to automatically shard
    # by data, and not by filename. This is because we are
    # not using multiple shard files in a way that is useful
    # to TensorFlow if the user wants to shard again later,
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    loaded_tfdata = loaded_tfdata.with_options(options)
    return loaded_tfdata


def _save_v1(
    *,
    dataset: tf.data.Dataset,
    tfdata_path: str,
    num_shards: int,
):
    """tfdata saver v1."""
    if num_shards != 1:
        raise DataBlobShardingValueError(
            "The ScalarStop DataBlob Persistence Protocol v1 only supports "
            f"num_shards=1. You passed {num_shards=}. Try saving with a "
            "higher protocol version."
        )
    return tf.data.experimental.save(
        dataset=dataset,
        path=tfdata_path,
        compression=None,
    )


def _save_v2(
    *,
    dataset: tf.data.Dataset,
    tfdata_path: str,
    num_shards: int,
):
    """tfdata saver v2."""
    return tf.data.experimental.save(
        dataset=dataset.enumerate(),
        path=tfdata_path,
        shard_func=make_num_shards_on_save(num_shards=num_shards),
        compression=None,
    )


def _save_v3(
    *,
    dataset: tf.data.Dataset,
    tfdata_path: str,
    num_shards: int,
):
    """
    tfdata saver v3.

    This implementation is identical to tfdata saver v2
    because the backwards-compatible changes are on the
    loading side.
    """
    return _save_v2(dataset=dataset, tfdata_path=tfdata_path, num_shards=num_shards)


def _save(
    dataset: tf.data.Dataset,
    tfdata_path: str,
    num_shards: int,
    save_load_version: int,
):
    """tfdata saver."""
    if save_load_version == 1:
        return _save_v1(
            dataset=dataset,
            tfdata_path=tfdata_path,
            num_shards=num_shards,
        )
    if save_load_version == 2:
        return _save_v2(
            dataset=dataset,
            tfdata_path=tfdata_path,
            num_shards=num_shards,
        )
    if save_load_version == 3:
        return _save_v3(
            dataset=dataset,
            tfdata_path=tfdata_path,
            num_shards=num_shards,
        )
    raise UnsupportedDataBlobSaveLoadVersion(version=save_load_version)


def tfdata_save(
    *,
    dataset: tf.data.Dataset,
    path: str,
    num_shards: int,
    save_load_version: int,
):
    """Save a tf.data dataset."""
    os.mkdir(path)

    # Save the tf.data Dataset.
    tfdata_path = os.path.join(path, _TFDATA_DIRECTORY_NAME)
    _save(
        dataset=dataset,
        tfdata_path=tfdata_path,
        num_shards=num_shards,
        save_load_version=save_load_version,
    )

    # Save the element spec.
    element_spec_path = os.path.join(path, _ELEMENT_SPEC_FILENAME)
    with open(element_spec_path, "wb") as fh:  # type: ignore
        scalarstop.pickle.dump(
            obj=dataset.element_spec,
            file=fh,
        )
