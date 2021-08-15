"""Private Keras callbacks that make :py:class:`~scalarstop.model.KerasModel` work."""
from typing import Any, Dict, Mapping, Optional

import tensorflow as tf


def logs_as_floats(logs: Mapping[str, Any]) -> Dict[str, float]:
    """Convert Keras metric log values to floats."""
    return {name: float(value) for name, value in logs.items()}


class BatchLoggingCallback(tf.keras.callbacks.Callback):
    """A Keras callback to handle some of the bookkeeping."""

    def __init__(
        self,
        *,
        scalarstop_model,
        logger,
    ):
        super().__init__()
        self._scalarstop_model = scalarstop_model
        self._logger = logger

    def on_train_batch_end(  # pylint: disable=signature-differs
        self, batch: int, logs: Dict[str, Any]
    ) -> None:
        """Enable issuing log messages at the end of every batch."""
        super().on_train_batch_end(batch=batch, logs=logs)
        float_logs = logs_as_floats(logs)
        self._logger.info(
            "Trained batch %s for epoch %s for model %s",
            batch,
            self._scalarstop_model.current_epoch,
            self._scalarstop_model.name,
            extra=dict(
                current_batch=batch,
                current_epoch=self._scalarstop_model.current_epoch,
                model_name=self._scalarstop_model.name,
                training_metrics=float_logs,
            ),
        )


class EpochCallback(tf.keras.callbacks.Callback):
    """A Keras callback to handle some of the bookkeeping."""

    def __init__(
        self,
        *,
        scalarstop_model,
        logger,
        models_directory: Optional[str] = None,
        train_store=None,
        log_epochs: bool = False,
    ):
        super().__init__()
        self._scalarstop_model = scalarstop_model
        self._models_directory = models_directory
        self._train_store = train_store
        self._log_epochs = log_epochs
        self._logger = logger

    def on_epoch_end(  # pylint: disable=signature-differs
        self, epoch: int, logs: Dict[str, Any]
    ) -> None:
        """
        Enable various tasks at the end of every epoch, such as:
         - saving the model to the filesystem.
         - saving epoch metrics to the TrainStore.
         - logging epoch metrics to a Python logger.
        """
        super().on_epoch_end(epoch=epoch, logs=logs)
        # Make sure that metrics are floats and not some
        # unserializable data type like tf.Tensor
        float_logs = logs_as_floats(logs)

        # Append epoch metrics to the model history.
        for metric, value in float_logs.items():
            if metric in self._scalarstop_model._history:
                self._scalarstop_model._history[metric].append(value)
            else:
                self._scalarstop_model._history[metric] = [value]

        # Save the model to the filesystem.
        if self._models_directory:
            self._scalarstop_model.save(self._models_directory)

        # Report the metric to the train store.
        if self._train_store:
            self._train_store.insert_model_epoch(
                epoch_num=self._scalarstop_model.current_epoch,
                model_name=self._scalarstop_model.name,
                metrics=float_logs,
                ignore_existing=True,
            )

        # Log the epoch.
        if self._log_epochs:
            self._logger.info(
                "Trained epoch %s for model %s",
                self._scalarstop_model.current_epoch,
                self._scalarstop_model.name,
                extra=dict(
                    current_epoch=self._scalarstop_model.current_epoch,
                    model_name=self._scalarstop_model.name,
                    training_metrics=float_logs,
                ),
            )
