"""Tools for storing metadata about :py:class:`~scalarstop.datablob.DataBlob` s."""
import json
import os
from typing import Any, Dict, Union

import scalarstop.pickle
from scalarstop._constants import _METADATA_JSON_FILENAME, _METADATA_PICKLE_FILENAME
from scalarstop.dataclasses import asdict
from scalarstop.exceptions import DataBlobNotFound
from scalarstop.hyperparams import HyperparamsType


class DataBlobMetadata:
    """
    Represents the metadata from a :py:class:`~scalarstop.datablob.DataBlob`
    that is saved to or loaded from the filesystem.

    When we save metadata to the filesystem, we save the same information
    in two files--one in JSON format and the other in Python Pickle format.
    The JSON file is human-readable and can be parsed by non-Python
    programs. The Pickle file is kept to ensure that the
    :py:class:`~scalarstop.datablob.DataBlob` 's hyperparams can be accurately
    deserialized--even if the hyperparams are not JSON-serializable.
    """

    @staticmethod
    def _pickle_filename(path: str) -> str:
        """
        The exact filename of the
        :py:class:`~scalarstop.datablob.DataBlob`'s Pickle metadata file."""
        return os.path.join(path, _METADATA_PICKLE_FILENAME)

    @staticmethod
    def _json_filename(path: str) -> str:
        """
        The exact filename of the
        :py:class:`~scalarstop.datablob.DataBlob` 's JSON metadata file.
        """
        return os.path.join(path, _METADATA_JSON_FILENAME)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "DataBlobMetadata":
        """
        Loads metadata from a :py:class:`~scalarstop.datablob.DataBlob` 's directory on the
        filesystem.
        """
        try:
            with open(cls._pickle_filename(path), "rb") as fh:
                metadata = scalarstop.pickle.load(fh)
        except FileNotFoundError as exc:
            raise DataBlobNotFound(path) from exc
        return cls(
            name=metadata["name"],
            group_name=metadata["group_name"],
            save_load_version=metadata.get("save_load_version", 1),
            num_shards=metadata.get("num_shards", 1),
            hyperparams=metadata["hyperparams"],
        )

    @classmethod
    def from_datablob(
        cls,
        datablob: "scalarstop.datablob.DataBlob",
        *,
        save_load_version: int,
        num_shards: int,
    ):
        """
        Creates a :py:class:`DataBlobMetadata` object in memory
        from a :py:class:`~scalarstop.datablob.DataBlob` instance.

        Args:
            datablob: The :py:class:`~scalarstop.datablob.DataBlob` for which this
                :py:class:`DataBlobMetadata` object is being created for.

            save_load_version: The protocol version used to save or load
                this :py:class:`~scalarstop.datablob.DataBlob` to/from the filesystem.

            num_shards: The number of shards to divide the
                :py:class:`~scalarstop.datablob.DataBlob` into when
                saving to the filesystem.
        """
        return cls(
            name=datablob.name,
            group_name=datablob.group_name,
            hyperparams=datablob.hyperparams,
            save_load_version=save_load_version,
            num_shards=num_shards,
        )

    def __init__(
        self,
        *,
        name: str,
        group_name: str,
        hyperparams: HyperparamsType,
        save_load_version: int,
        num_shards: int,
    ):
        """
        Creates a :py:class:`DataBlobMetadata` object in memory.

        Args:
            name: The :py:class:`~scalarstop.datablob.DataBlob` name.

            group_name: The :py:class:`~scalarstop.datablob.DataBlob` group name.

            hyperparams: The ``Hyperparams`` object for the
                :py:class:`~scalarstop.datablob.DataBlob`.
                This has to be an instance of
                :py:class:`~scalarstop.hyperparams.HyperparamsType` and
                **not** a Python dictionary.

            save_load_version: The protocol version used to save or load
                this :py:class:`~scalarstop.datablob.DataBlob` to/from the filesystem.

            num_shards: The number of shards to divide the
                :py:class:`~scalarstop.datablob.DataBlob` into when
                saving to the filesystem.
        """
        if not isinstance(hyperparams, HyperparamsType):
            raise TypeError(
                "ScalarStop's DataBlobMetadata requires a HyperparamsType "
                "instance for the `hyperparams` parameter. You provided "
                f"the object {hyperparams} of type {type(hyperparams)}."
            )
        self.name = name
        self.group_name = group_name
        self.save_load_version = save_load_version
        self.num_shards = num_shards
        self.hyperparams = hyperparams

    def to_dict(self, *, hyperparams_as_dict: bool = False) -> Dict[str, Any]:
        """
        Return the metadata as a Python dictionary.

        Args:
            hyperparams_as_dict: Set to ``True`` to convert a
            :py:class:`~scalarstop.hyperparams.HyperparamsType`
            object to a Python dictionary.
        """
        if hyperparams_as_dict:
            hyperparams: Union[HyperparamsType, Dict[str, Any]] = asdict(
                self.hyperparams
            )
        else:
            hyperparams = self.hyperparams
        return dict(
            name=self.name,
            group_name=self.group_name,
            save_load_version=self.save_load_version,
            num_shards=self.num_shards,
            hyperparams=hyperparams,
        )

    def save(self, path: str):
        """
        Save the metadata to a given
        :py:class:`~scalarstop.datablob.DataBlob` directory on
        the filesystem.

        Args:
            path: The :py:class:`~scalarstop.datablob.DataBlob` directory
                on the filesystem to save the metadata to.
        """
        with open(self._json_filename(path), "w", encoding="utf-8") as fh:
            json.dump(
                obj=self.to_dict(hyperparams_as_dict=True),
                fp=fh,
                sort_keys=True,
                indent=4,
            )
        with open(self._pickle_filename(path), "wb") as fh:  # type: ignore
            scalarstop.pickle.dump(
                obj=self.to_dict(hyperparams_as_dict=False),
                file=fh,
            )

    def __eq__(self, other) -> bool:
        if isinstance(other, DataBlobMetadata):
            self_dict = self.to_dict(hyperparams_as_dict=False)
            other_dict = other.to_dict(hyperparams_as_dict=False)
            return self_dict == other_dict
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
