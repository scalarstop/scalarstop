"""
Wrappers that specify models trained on specific
:py:class:`~scalarstop.datablob.DataBlob` instances.

Creating and training models
----------------------------

The purpose of a :py:class:`Model` subclass instance--such as
:py:class:`KerasModel`--is to join together a
:py:class:`~scalarstop.datablob.DataBlob` instance
and :py:class:`~scalarstop.model_template.ModelTemplate` instance
into a trained model.

It also manages saving and loading models to/from the filesystem
and save hyperparameters and training metrics to the
:py:class:`~scalarstop.train_store.TrainStore`.

The `ScalarStop Tutorial <https://nbviewer.jupyter.org/github/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_
demonstrates how to
use ScalarStop when training real models on real data. Below
is a brief sketch of how to load, save, and train models.

First, we subclass :py:class:`~scalarstop.datablob.DataBlob` and
create an instance. This is where we store our training, validation,
and test sets.

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

And when we create an instance of our
:py:class:`~scalarstop.datablob.DataBlob` subclass, we should batch
it if we plan on training a model with it.

>>> datablob = MyDataBlob(hyperparams=dict(cols=3)).batch(2)

Then, we define the *architecture* of the model we want to train
by subclassing :py:class:`~scalarstop.model_template.ModelTemplate`
and creating an instance.

>>> class MyModelTemplate(sp.ModelTemplate):
...    @sp.dataclass
...
...    class Hyperparams(sp.HyperparamsType):
...        hidden_units: int
...        optimizer: str = "adam"
...
...    def new_model(self):
...        model = tf.keras.Sequential(
...            layers=[
...                tf.keras.layers.Dense(
...                    units=self.hyperparams.hidden_units,
...                    activation="relu",
...                ),
...                tf.keras.layers.Dense(
...                    units=1,
...                    activation="sigmoid"
...                ),
...           ],
...            name=self.name,
...        )
...        model.compile(
...            optimizer=self.hyperparams.optimizer,
...            loss="binary_crossentropy",
...            metrics=["accuracy"],
...        )
...        return model
>>> model_template = MyModelTemplate(hyperparams=dict(hidden_units=5))

Now we create a :py:class:`KerasModel` instance that bridges together
our :py:class:`~scalarstop.datablob.DataBlob` and
:py:class:`~scalarstop.model_template.ModelTemplate` instances.

We'll also pass a directory to ``models_directory``. If we have
a model saved in a subdirectory of ``models_directory``, we'll
load that model instead of starting from scratch.

>>> import os
>>> import tempfile
>>> tempdir = tempfile.TemporaryDirectory()
>>>
>>> model = sp.KerasModel.from_filesystem_or_model_template(
...    datablob=datablob,
...    model_template=model_template,
...    models_directory=tempdir.name,
... )

Then you can call :py:meth:`KerasModel.fit` to fit your new model using your
:py:class:`~scalarstop.datablob.DataBlob` 's training and validation sets.
We pass ``models_directory`` here again--this time to *save* our
model in a subdirectory.

>>> history = model.fit(final_epoch=3, verbose=0, models_directory=tempdir.name)

You can call :py:meth:`KerasModel.evalute` to evaluate your
model against your :py:class:`~scalarstop.datablob.DataBlob` 's
test set--or another :py:class:`tf.data.Dataset`
of your choosing.

>>> test_set_metrics = model.evaluate(verbose=0)

(And now we clean up the temporary directory from our example.)
>>> tempdir.cleanup()

Using the :py:class:`~scalarstop.train_store.TrainStore`
--------------------------------------------------------

If you pass a :py:class:`~scalarstop.train_store.TrainStore` to
:py:meth:`KerasModel.fit`, then the metrics generated while
training will be saved to the Train Store's database, along with
the :py:class:`~scalarstop.datablob.DataBlob` and
:py:class:`~scalarstop.model_template.ModelTemplate`
names and hyperparameters.

"""  # pylint: disable=line-too-long
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import tensorflow as tf

from scalarstop._filesystem import rmtree
from scalarstop.datablob import DataBlob
from scalarstop.exceptions import IsNotImplemented, ModelNotFoundError
from scalarstop.model_template import ModelTemplate
from scalarstop.train_store import TrainStore
from scalarstop.warnings import warn

_LOGGER = logging.getLogger(__name__)

_HISTORY_FILENAME = "history.json"


class Model:
    """Abstract parent class for all ScalarStop models."""

    _this_checkpoint_directory: Optional[str] = None

    @classmethod
    def from_filesystem(
        cls, *, datablob: DataBlob, model_template: ModelTemplate, models_directory: str
    ):
        """
        Load an already-trained model from the filesystem.

        Args:
            datablob: The :py:class:`~scalarstop.models.DataBlob`
                used to train the model that we are looking for.

            model_template: The :py:class:`~scalarstop.ModelTemplate`
                used to create the model that we are looking for.

            models_directory: The directory where you store all of your
                pretrained models. This is the *parent* directory
                of a single pretrained model.

        Returns:
            A :py:class:`Model` with weights and configuration from
                the filesystem.

        Raises:
            ModelNotFoundError: Raised when we cannot find the model.
                If you intend on subclassing
                :py:meth:`~Model.from_filesystem`, make sure to raise
                this exception when you cannot find the model.
        """
        raise IsNotImplemented(f"{cls.__name__}.from_filesystem()")

    @classmethod
    def from_model_template(cls, *, datablob: DataBlob, model_template: ModelTemplate):
        """
        Create a new :py:class:`Model` from a :py:class:`~scalarstop.ModelTemplate`.

        Args:
            datablob: The :py:class:`~scalarstop.models.DataBlob`
                that we will use to train the model.

            model_template: The :py:class:`~scalarstop.ModelTemplate`
                that we will use to create the model.

        Returns:
            A brand-new :py:class:`Model` that has not yet been trained.
        """
        return cls(
            datablob=datablob,
            model_template=model_template,
            model=model_template.new_model(),
        )

    @classmethod
    def from_filesystem_or_model_template(
        cls, *, datablob: DataBlob, model_template: ModelTemplate, models_directory: str
    ):
        """
        Load a saved model from the filesystem. If we can't find one, create a new one with
        the supplied :py:class:`~scalarstop.ModelTemplate`.

        Args:
            datablob: The :py:class:`~scalarstop.models.DataBlob`
                that we will use to train the model.

            model_template: The :py:class:`~scalarstop.ModelTemplate`
                that we will use to create the model.

            models_directory: The directory where you store all of your
                pretrained models. This is the *parent* directory
                of a single pretrained model.

        Returns:
            A :py:class:`Model` instance.
        """
        try:
            return cls.from_filesystem(
                datablob=datablob,
                model_template=model_template,
                models_directory=models_directory,
            )
        except ModelNotFoundError:
            return cls.from_model_template(
                datablob=datablob, model_template=model_template
            )

    def __init__(self, *, datablob: DataBlob, model_template: ModelTemplate, model):
        self._datablob = datablob
        self._model_template = model_template
        self._model = model
        self._name = self._make_name(
            datablob_name=self._datablob.name,
            model_template_name=self._model_template.name,
        )

    @staticmethod
    def _make_name(*, model_template_name: str, datablob_name: str):
        """
        Create a model name from a
        :py:class:`~scalarstop.ModelTemplate`
        name and a :py:class:`~scalarstop.DataBlob` name.
        """
        return f"mt_{model_template_name}__d_{datablob_name}"

    @property
    def name(self):
        """
        This model's name.

        If you intend on overriding this method, you should make sure
        that two :py:class:`Model` s trained on the same
        :py:class:`~scalarstop.DataBlob` and
        :py:class:`~scalarstop.ModelTemplate` have the
        same name.
        """
        return self._name

    @property
    def datablob(self):
        """
        Returns the :py:class:`~scalarstop.DataBlob`
        used to create this model.
        """
        return self._datablob

    @property
    def model_template(self):
        """
        Returns the :py:class:`~scalarstop.ModelTemplate`
        used to create this model.
        """
        return self._model_template

    @property
    def model(self):
        """The model object from the underlying machine learning framework."""
        return self._model

    @staticmethod
    def load(model_path: str):
        """
        Loads a model.

        Raises:
            ModelNotFoundError: Raised when a saved copy of the model cannot
                be found at the given ``directory``. If you are overriding
                this method, you should make sure to catch any exceptions
                your code generates, such as :py:class:`FileNotFoundError`,
                and re-reraise them as
                :py:class:`~scalarstop.exceptions.ModelNotFoundError`.
        """
        raise IsNotImplemented("Model.load()")

    @property
    def history(self) -> Mapping[str, Sequence[float]]:
        """Returns the per-epoch history for training and validation metrics."""
        raise IsNotImplemented(f"{self.__class__.__name__}.history")

    @property
    def current_epoch(self):
        """Returns how many epochs the current model has been trained."""
        raise IsNotImplemented(f"{self.__class__.__name__}.current_epoch")

    def save(self, models_directory: str):
        """Saves a model to the given directory."""
        raise IsNotImplemented(f"{self.__class__.__name__}.save()")

    def fit(self, *, final_epoch: int, **kwargs):
        """
        Fits the given model to the given
        :py:class:`~scalarstop.DataBlob`.
        """
        raise IsNotImplemented(f"{self.__class__.__name__}.fit()")

    def predict(self, dataset):
        """Runs predictions with the dataset on the model."""
        raise IsNotImplemented(f"{self.__class__.__name__}.predict()")

    def evaluate(self, dataset=None):
        """Evaluate the model on a dataset."""
        raise IsNotImplemented(f"{self.__class__.__name__}.evaluate()")


_KERAS_HISTORY_TYPE = Dict[str, List[float]]


def _logs_as_floats(logs: Mapping[str, Any]):
    """Convert Keras metric log values to floats."""
    return {name: float(value) for name, value in logs.items()}


class _ScalarStopKerasCallback(tf.keras.callbacks.Callback):
    """A Keras callback to handle some of the bookkeeping."""

    def __init__(
        self,
        *,
        scalarstop_model: "KerasModel",
        models_directory: Optional[str] = None,
        train_store: Optional[TrainStore] = None,
        log_epochs: bool = False,
        logger=None,
    ):
        super().__init__()
        self._scalarstop_model = scalarstop_model
        self._models_directory = models_directory
        self._train_store = train_store
        self._log_epochs = log_epochs
        self._logger = logger or _LOGGER

    def on_epoch_end(self, epoch: int, logs=None):
        super().on_epoch_end(epoch=epoch, logs=logs)
        # Make sure that metrics are floats and not some
        # unserializable data type like tf.Tensor
        float_logs = _logs_as_floats(logs)

        # Append epoch metrics to the model history.
        for metric, value in float_logs.items():
            if metric in self._scalarstop_model._history:
                self._scalarstop_model._history[metric].append(value)
            else:
                self._scalarstop_model._history[metric] = [value]

        # Save the model to the filesystem.
        if self._models_directory:
            self._scalarstop_model.save(self._models_directory)

        if self._train_store:
            self._train_store.insert_model_epoch(
                epoch_num=self._scalarstop_model.current_epoch,
                model_name=self._scalarstop_model.name,
                metrics=float_logs,
                ignore_existing=True,
            )

        # Log the epoch.
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


class KerasModel(Model):
    """Trains :py:mod:`tf.keras` machine learning models generated by a :py:class:`~scalarstop.model_template.ModelTemplate` on the training and validation sets in a :py:class:`~scalarstop.datablob.DataBlob`."""  # pylint: disable=line-too-long

    @classmethod
    def from_filesystem(
        cls,
        *,
        datablob: DataBlob,
        model_template: ModelTemplate,
        models_directory: str,
    ):
        model_name = cls._make_name(
            datablob_name=datablob.name, model_template_name=model_template.name
        )
        model_path = os.path.join(models_directory, model_name)

        # Load the model.
        try:
            model = tf.keras.models.load_model(model_path)
        except (OSError, IOError) as exc:
            raise ModelNotFoundError(model_path) from exc

        # Try to load the history.
        history_path = os.path.join(model_path, _HISTORY_FILENAME)
        try:
            with open(history_path, "r") as fh:
                history = json.load(fh)
        except FileNotFoundError:
            warn(
                "Tried and failed to load Keras model "
                f"history at {history_path} . Will load model without history."
            )
            history = None

        # Come up with the model name.
        return cls(
            datablob=datablob,
            model_template=model_template,
            model=model,
            history=history,
        )

    def __init__(
        self,
        *,
        datablob: DataBlob,
        model_template: ModelTemplate,
        model,
        history: Optional[_KERAS_HISTORY_TYPE] = None,
    ):
        super().__init__(datablob=datablob, model_template=model_template, model=model)

        self._history: Dict[str, List[float]] = history or dict()

        # If the model does not have a valid input shape, then we build it
        # with the DataBlob training element_spec.
        try:
            self._model.input_shape
        except AttributeError:
            x_spec, _ = self._datablob.training.element_spec
            self._model.build(input_shape=x_spec.shape)

    def __repr__(self) -> str:
        if self.current_epoch == 1:
            epoch_str = "epoch"
        else:
            epoch_str = "epochs"
        return f"<sp.KerasModel {self.name} ({self.current_epoch} {epoch_str})>"

    @property
    def history(self) -> Mapping[str, Sequence[float]]:
        """Returns the history for the Keras model."""
        return self._history

    @property
    def current_epoch(self):
        if "loss" in self.history:
            return len(self.history["loss"])
        return 0

    def save(self, models_directory: str):
        model_path = os.path.join(models_directory, self.name)
        try:
            self._model.save(
                filepath=model_path,
                overwrite=True,
                include_optimizer=True,
                save_format="tf",
            )
            history_path = os.path.join(model_path, _HISTORY_FILENAME)
            with open(history_path, "w") as fp:
                json.dump(
                    obj=self.history,
                    fp=fp,
                    sort_keys=True,
                    indent=4,
                )
        except BaseException:
            warn(
                "Caught exception while saving Keras model. "
                f"Removing partially-saved results at {model_path}"
            )
            rmtree(model_path)
            raise

    def fit(  # pylint: disable=arguments-differ
        self,
        *,
        final_epoch: int,
        verbose: Optional[int] = None,
        models_directory: Optional[str] = None,
        log_epochs: bool = False,
        callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
        train_store: Optional[TrainStore] = None,
        **kwargs,
    ):
        """
        Fit the Keras model to the :py:class:`~scalarstop.DataBlob`
        that this model was created for.

        Args:
            final_epoch: The epoch number *to train to*. If the model
                has already been trained for ``final_epoch`` or more epochs,
                then this function will do nothing. This helps make
                training a machine learning model into an idempotent operation.

            verbose: The verbosity to level to use.

            models_directory: The directory to save this machine learning model
                every epoch.

            log_epochs: Emit a Python logging message as an ``INFO`` level
                log for every single epoch.

            train_store: A :py:class:`~scalarstop.TrainStore`
                instance, which is a client that persists metadata about
                :py:class:`~scalarstop.DataBlob` s,
                :py:class:`~scalarstop.ModelTemplate` s,
                and :py:class:`~scalarstop.model.Model` s.

            callbacks: A list of Keras callbacks to use while training.
        """
        if kwargs:
            raise ValueError(
                f"Unknown arguments to {self.__class__.__name__}.fit(): {kwargs}"
            )

        if final_epoch > self.current_epoch:
            if verbose is None:
                verbose = 1

            if callbacks:
                callbacks = list(callbacks)
            else:
                callbacks = []

            callbacks.append(
                _ScalarStopKerasCallback(
                    scalarstop_model=self,
                    models_directory=models_directory,
                    log_epochs=log_epochs,
                    train_store=train_store,
                )
            )

            if train_store:
                train_store.insert_datablob(self._datablob, ignore_existing=True)
                train_store.insert_model_template(
                    self._model_template, ignore_existing=True
                )
                train_store.insert_model(self, ignore_existing=True)

            self._model.fit(
                x=self._datablob.training,
                validation_data=self._datablob.validation,
                verbose=verbose,
                callbacks=callbacks,
                initial_epoch=self.current_epoch,
                epochs=final_epoch,
            )
        return self.history

    def predict(  # pylint: disable=arguments-differ
        self,
        dataset,
        verbose: Optional[int] = None,
        callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
    ):
        """
        Use the model to generate predictions on this dataset.

        Args:
            dataset: An input dataset to predict on. This accepts
                any type type that :py:class:`tf.keras.Model` can
                generate predictions for.

            verbose: Verbosity level for predictions.

            callbacks: A list of Keras callbacks to use while
                making predictions.
        """
        if verbose is None:
            verbose = 1

        if callbacks:
            callbacks = list(callbacks)
        else:
            callbacks = []
        return self._model.predict(x=dataset, verbose=verbose, callbacks=callbacks)

    def evaluate(  # pylint: disable=arguments-differ
        self,
        dataset=None,
        verbose: Optional[int] = None,
        callbacks: Optional[Sequence[tf.keras.callbacks.Callback]] = None,
    ):
        """
        Evaluate this model on the :py:class:`~scalarstop.DataBlob`'s test set.

        Optionally, you can provide another :py:class:`tf.data.Dataset` via the
        ``dataset`` parameter.

        Args:
            dataset: Another :py:class:`tf.data.Dataset` to evalaute instead of
                the test set of the provided :py:class:`~scalarstop.DataBlob`.

            verbose: Specifiy verbosity for evaluating this model.

            callbacks: A list of Keras callbacks to use when evaluating the model.
        """
        if dataset is None:
            dataset = self._datablob.test

        if verbose is None:
            verbose = 1

        if callbacks:
            callbacks = list(callbacks)
        else:
            callbacks = []
        return self._model.evaluate(x=dataset, verbose=verbose, callbacks=callbacks)
