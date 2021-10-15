"""Helpers for TensorFlow's ``"TF_CONFIG"`` environment variable."""
import json
import os
from typing import Any, Dict, Optional


class TFConfig:
    """
    A helper object for querying the TensorFlow configuration in the
    ``"TF_CONFIG"`` environment variable.
    """

    def __init__(self, *, tf_config_str: str = ""):
        """
        Args:
            tf_config_str: A string containing a value for the
                ``"TF_CONFIG"`` environment variable. This argument
                is typically used for testing.
        """
        if not tf_config_str:
            tf_config_str = os.environ.get("TF_CONFIG", "{}")
        self._tf_config = json.loads(tf_config_str)
        task = self._tf_config.get("task", {})
        self._task_type = task.get("type")
        self._task_index = task.get("index", 0)
        self._is_chief = (
            self._task_type is None
            or self._task_type == "chief"
            or (self._task_type == "worker" and self._task_index == 0)
        )

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Returns a Python dictionary containing a ``"TF_CONFIG"`` configuration."""
        return self._tf_config

    def num_nodes(self, task_type: str) -> int:
        """
        Returns the number of nodes belonging to the task type.

        Possible task types include:
         * ``"chief"``
         * ``"worker"``
         * ``"ps"``
         * ``"evaluator"``

        Or you can pass ``task_type = "ALL"`` to count tasks of all types.

        Args:
            task_type: The task type for a worker.

        Returns:
            Returns the number of nodes for each type.
        """
        if task_type == "all":
            return sum(
                [
                    len(nodes_by_type)
                    for nodes_by_type in self._tf_config.get("cluster", {}).values()
                ]
            )
        return len(self._tf_config.get("cluster", {}).get(task_type, ()))

    @property
    def task_type(self) -> Optional[str]:
        """
        Returns the current process's task type.

        Possible task types include:
         * ``None``
         * ``"chief"``
         * ``"worker"``
         * ``"ps"``
         * ``"evaluator"``
        """
        return self._task_type

    @property
    def task_index(self) -> int:
        """
        Returns the current process's task index.

        The task index is a value between ``0`` and ``n-1``,
        where ``n`` is the number of task processes in your
        TensorFlow cluster.

        This returns ``0`` if the ``"TF_CONFIG"`` environment
        variable is not configured.
        """
        return self._task_index

    @property
    def is_chief(self) -> bool:
        """
        Returns ``True`` if the current process is the chief node in the cluster.

        This method will also return ``True`` if the ``"TF_CONFIG"``
        environment variable is not configured, as that would suggest
        that this process the chief node--and the *only* node--in
        the cluster.
        """
        return self._is_chief
