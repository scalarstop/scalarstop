"""
Every custom ScalarStop exception.
"""

from typing import Any, Optional


class ScalarStopException(Exception):
    """
    Base class for all ScalarStop exceptions.
    """


class InconsistentCachingParameters(ValueError, ScalarStopException):
    """
    Raised when DataBlob.cache() is set with inconsistent parameters.

    An example of setting inconsistent parameters is when the user says
    ``training=False`` and ``preload_training=True``.
    """


class YouForgotTheHyperparams(ValueError, ScalarStopException):
    """Raised when the user creates a class, but forgets to add a nested Hyperparams class."""

    def __init__(self, class_name: str):
        super().__init__(
            "Please make sure to define a dataclass containing "
            f"hyperparameters at `{class_name}.Hyperparams`. "
            "Your dataclass should subclass `sp.HyperparamsType` "
            "and is decorated with a `@sp.dataclass` decorator."
        )


class WrongHyperparamsType(TypeError, ScalarStopException):
    """Raised when the user passes an object with the wrong type as hyperparams."""

    def __init__(self, *, hyperparams: Any, class_name: str):
        super().__init__(
            f"You passed an object with the type {type(hyperparams)} as hyperparams. "
            "Check with the surrounding class. You should either pass a dictionary "
            "or other mapping-like object or directly pass "
            f"a {class_name}.Hyperparams object."
        )


class DataBlobNotFound(FileNotFoundError, ScalarStopException):
    """
    Raised when we cannot load a :py:class:`DataBlob`
    from the filesystem.
    """

    def __init__(self, path: str):
        """
        Args:
            path: The directory that we went
                looking for the :py:class:`DataBlob` when we
                were not able to find it.
        """
        super().__init__("Could not load a DataBlob from the filesystem at " + path)


class ModelNotFoundError(FileNotFoundError, ScalarStopException):
    """Raised when we cannot find a saved model."""

    def __init__(self, directory: str):
        super().__init__(f"Could not find a saved model at {directory}.")


class FileExists(FileExistsError, ScalarStopException):
    """
    Raise when we want to save a :py:class:`DataBlob` but a
    file exists at the save path
    """


class FileExistsDuringDataBlobCreation(FileExists):
    """
    Raised when a file exists at the save path, but we think the
    file was created *while* we were saving the :py:class:`DataBlob`,
    which suggests that the user's code has a race conditon.
    """


class TensorFlowDatasetNotFound(FileNotFoundError, ScalarStopException):
    """
    Raised when we canno load a :py:class:`tf.data.Dataset`
    from the filesystem.
    """

    def __init__(self, directory: str):
        super().__init__(f"Could not load a tf.data.Dataset from {directory}")


class ElementSpecNotFound(FileNotFoundError, ScalarStopException):
    """
    Raised when we try to guess the location of a serialized
    ``element_spec`` for a :py:class:`tf.data.Dataset`.
    """

    def __init__(self, directory: str):
        """
        Args:
            directory: The directory where we were looking for
                the ``element_spec`` for a :py:class:`tf.data.Dataaset`.
                The TensorFlow API assumes that you have the
                ``element_spec`` on hand, but ScalarStop likes to
                save it to disk.

        """
        super().__init__(
            "Could not find an element_spec for the tf.data.Dataset "
            f"at {directory}. Either check your path or provide an "
            "element_spec."
        )


class IsNotImplemented(NotImplementedError, ScalarStopException):
    """
    Raised when someone calls a function that isn't yet implemented.
    """

    def __init__(self, name: str):
        """
        Args:
            name: The name of the function, method, or property that
                was not implemented. This is useful in case the stack
                trace is hard to read for some reason.
        """
        super().__init__(f"Make sure to implement `{name}` in a subclass.")


class SQLite_JSON_ModeDisabled(RuntimeError, ScalarStopException):
    """
    Raised when the user is running the
    :py:class:`~scalarstop.TrainStore` backed by
    a SQLite database, but the SQLite JSON1 extension is
    unavailable.
    """

    def __init__(self) -> None:
        super().__init__(
            "You want to start the TrainStore with a SQLite "
            "database, but we could not find the SQLite JSON1 "
            "extension in your Python installation."
        )


class UnsupportedDataBlobSaveLoadVersion(ScalarStopException):
    """Raised when we don't recognize the Load/Save version number."""

    def __init__(self, *, version: int):
        super().__init__(
            f"The DataBlob Load/Save version `{version}` is unsupported in this "
            "version of ScalarStop."
        )


class DataBlobShardingNotSupported(ScalarStopException):
    """
    Raised when the user tries to shard a saved DataBlob
    and the Load/Save version is too old.
    """

    def __init__(
        self,
        *,
        version: int,
        offset: Optional[int],
        quantity: int,
        total_num_shards: int,
    ):
        super().__init__(
            "Reading saved DataBlobs by shard is not supported "
            f"in this DataBlob Load/Save {version=}. "
            f"You requested sharding parameters "
            f"{offset=} {quantity=} {total_num_shards=}."
        )


class DataBlobShardingValueError(ValueError, ScalarStopException):
    """Raised when sharding parameters are not correct."""
