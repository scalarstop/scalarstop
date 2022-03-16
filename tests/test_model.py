"""Unit tests for scalarstop.model."""
import doctest
import logging
import os
import tempfile
import unittest
import unittest.mock
import warnings

import tensorflow as tf

import scalarstop as sp
from tests.assertions import (
    assert_directory,
    assert_keras_saved_model_directory,
    assert_model_after_fit,
    assert_spkeras_models_are_equal,
)
from tests.fixtures import (
    MyDataBlob,
    MyDataBlobRepeating,
    MyModelTemplate,
    requires_sqlite_json,
)


def load_tests(loader, tests, ignore):  # pylint: disable=unused-argument
    """Have the unittest loader also run doctests."""
    tests.addTests(doctest.DocTestSuite(sp.model))
    return tests


class TestModel(unittest.TestCase):
    """Test Model."""

    def test_not_implemented(self):
        """Assert that various methods need to be implemented in a subclass."""
        datablob = MyDataBlob(hyperparams=dict(rows=10, cols=5)).batch(2)
        model_template = MyModelTemplate(hyperparams=dict(layer_1_units=2))

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            sp.Model.from_filesystem(
                datablob=datablob, model_template=model_template, models_directory="/"
            )

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            sp.Model.load("/")

        model = sp.Model(datablob=datablob, model_template=model_template)

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.history  # pylint: disable=pointless-statement

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.current_epoch  # pylint: disable=pointless-statement

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.save("a")

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.fit(final_epoch=1)

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.predict(dataset=tf.data.Dataset.from_tensor_slices([1, 2]))

        with self.assertRaises(sp.exceptions.IsNotImplemented):
            model.evaluate(dataset=tf.data.Dataset.from_tensor_slices([1, 2]))


class TestKerasModel(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Test KerasModel."""

    def setUp(self):
        self.temp_dir_context = (
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )
        self.models_directory = self.temp_dir_context.name
        self.datablob = MyDataBlob(hyperparams=dict(rows=10, cols=5)).batch(2)
        self.model_template = MyModelTemplate(hyperparams=dict(layer_1_units=2))
        self.keras_model = sp.KerasModel(
            datablob=self.datablob,
            model_template=self.model_template,
        )

    def tearDown(self):
        self.temp_dir_context.cleanup()

    def test_model_not_found(self):
        """Test what happens when we try to load a nonexistent model."""
        with self.assertRaises(sp.exceptions.ModelNotFoundError):
            sp.KerasModel.from_filesystem(
                datablob=self.datablob,
                model_template=self.model_template,
                models_directory="",
            )
        with self.assertRaises(sp.exceptions.ModelNotFoundError):
            sp.KerasModel.from_filesystem(
                datablob=self.datablob,
                model_template=self.model_template,
                models_directory=self.models_directory,
            )

    def test_model_not_found_missing_history(self):
        """Test what happens when we load a KerasModel that has no history."""
        # Fit a model.
        retval = self.keras_model.fit(final_epoch=2, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=self.keras_model, expected_epochs=2
        )

        # Use Keras's save function, which will not save a history.json.
        self.keras_model.model.save(
            filepath=os.path.join(self.models_directory, self.keras_model.name),
            overwrite=True,
            include_optimizer=True,
            save_format="tf",
        )

        # Try to load it back.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded_model = sp.KerasModel.from_filesystem(
                datablob=self.datablob,
                model_template=self.model_template,
                models_directory=self.models_directory,
            )
        assert_model_after_fit(return_value={}, model=loaded_model, expected_epochs=0)

    def test_model_already_trained(self):
        """
        Test what happens when we train to a final epoch less
        than the number of epochs already trained.
        """
        # Assert that the model has not been trained.
        assert_model_after_fit(
            return_value={}, model=self.keras_model, expected_epochs=0
        )

        # Train for 3 epochs.
        retval = self.keras_model.fit(final_epoch=3, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=self.keras_model, expected_epochs=3
        )

        # Calling fit() again should have no effect.
        retval = self.keras_model.fit(final_epoch=3, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=self.keras_model, expected_epochs=3
        )

        # Also should have no effect.
        retval = self.keras_model.fit(final_epoch=2, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=self.keras_model, expected_epochs=3
        )

        # This should train for 2 more epochs.
        retval = self.keras_model.fit(final_epoch=5, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=self.keras_model, expected_epochs=5
        )

        # Save and load back. Assert the model is the same.
        self.keras_model.save(self.models_directory)
        loaded_model_1 = sp.KerasModel.from_filesystem(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        assert_spkeras_models_are_equal(self.keras_model, loaded_model_1)
        assert_model_after_fit(
            return_value=retval, model=loaded_model_1, expected_epochs=5
        )

        # Calling fit() should have no effect.
        retval = loaded_model_1.fit(final_epoch=5, verbose=0)
        assert_model_after_fit(
            return_value=retval, model=loaded_model_1, expected_epochs=5
        )

    def test_fit_works(self):
        """Test that fitting a model works."""
        history = self.keras_model.fit(final_epoch=2, verbose=0)
        self.assertEqual(
            sorted(history.keys()),
            [
                "binary_accuracy",
                "loss",
                "precision",
                "recall",
                "val_binary_accuracy",
                "val_loss",
                "val_precision",
                "val_recall",
            ],
        )
        assert_model_after_fit(
            return_value=history, model=self.keras_model, expected_epochs=2
        )

    def test_from_filesystem_or_new(self):
        """
        Test that we can load models from the filesystem when it exists,
        and that we fallback to creating a new model.
        """
        # Create two brand new models and assert that they've been untrained.
        model1 = sp.KerasModel.from_filesystem_or_new(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        model2 = sp.KerasModel.from_filesystem_or_new(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        assert_model_after_fit(return_value={}, model=model1, expected_epochs=0)
        assert_model_after_fit(return_value={}, model=model2, expected_epochs=0)
        assert_spkeras_models_are_equal(model1, model2)

        # Assert that we have no saved models.
        assert_directory(self.models_directory, [])

        # Fit the first model and verify that it has been saved.
        model1_history = model1.fit(
            final_epoch=2, verbose=0, models_directory=self.models_directory
        )
        assert_directory(self.models_directory, [model1.name])

        # Assert that our other model has been unchanged.
        assert_model_after_fit(return_value={}, model=model2, expected_epochs=0)

        # Create a new model. This time it should be loaded from the filesystem.
        model3 = sp.KerasModel.from_filesystem_or_new(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        assert_model_after_fit(
            return_value=model1_history, model=model3, expected_epochs=2
        )
        assert_spkeras_models_are_equal(model1, model3)

    def test_save_and_load(self):
        """Test that we can save and load Keras models."""
        # Test that we can save a model before doing any training.
        self.keras_model.save(self.models_directory)
        assert_directory(self.models_directory, [self.keras_model.name])
        model_path = os.path.join(self.models_directory, self.keras_model.name)
        assert_keras_saved_model_directory(model_path)

        # Test loading the model we saved.
        loaded_model_1 = sp.KerasModel.from_filesystem(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        self.assertEqual(self.keras_model.name, loaded_model_1.name)
        assert_spkeras_models_are_equal(self.keras_model, loaded_model_1)

        # Test fitting the model and saving again.
        loaded_model_1.fit(final_epoch=3, verbose=0)
        self.assertEqual(loaded_model_1.current_epoch, 3)
        loaded_model_1.save(self.models_directory)
        loaded_model_1.fit(final_epoch=3, verbose=0)
        # Load the model from the filesystem again.
        loaded_model_2 = sp.KerasModel.from_filesystem(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        assert_directory(self.models_directory, [loaded_model_2.name])
        assert_keras_saved_model_directory(model_path)
        assert_spkeras_models_are_equal(loaded_model_1, loaded_model_2)

        # Fit the model again and save.
        loaded_model_2.fit(final_epoch=5, verbose=0)
        self.assertEqual(loaded_model_2.current_epoch, 5)
        loaded_model_2.save(self.models_directory)

        # And load it again.
        loaded_model_3 = sp.KerasModel.from_filesystem(
            datablob=self.datablob,
            model_template=self.model_template,
            models_directory=self.models_directory,
        )
        assert_directory(self.models_directory, [loaded_model_3.name])
        assert_keras_saved_model_directory(model_path)
        assert_spkeras_models_are_equal(loaded_model_2, loaded_model_3)

    def test_save_callback_with_different_directories(self):
        """Test that the save callback works."""
        # Train the first model and save it.
        model1 = self.keras_model
        model_name = model1.name
        with tempfile.TemporaryDirectory() as dir1:
            model1.fit(final_epoch=3, verbose=0, models_directory=dir1)
            self.assertEqual(model1.current_epoch, 3)
            assert_directory(dir1, [model_name])
            assert_keras_saved_model_directory(os.path.join(dir1, model_name))

            # Load the model from the filesystem, train it, and save it.
            model2 = sp.KerasModel.from_filesystem(
                datablob=self.datablob,
                model_template=self.model_template,
                models_directory=dir1,
            )
            self.assertEqual(model2.current_epoch, 3)
        assert_spkeras_models_are_equal(model1, model2)
        with tempfile.TemporaryDirectory() as dir2:
            model2.fit(final_epoch=6, verbose=0, models_directory=dir2)

            # Load the model from the filesystem again... train it... and save it.
            model3 = sp.KerasModel.from_filesystem(
                datablob=self.datablob,
                model_template=self.model_template,
                models_directory=dir2,
            )
            self.assertEqual(model3.current_epoch, 6)
        assert_spkeras_models_are_equal(model2, model3)

    def test_predict(self):
        """Test that KerasModel.predict() works."""
        something = self.keras_model.predict(dataset=self.datablob.training, verbose=0)
        self.assertEqual(len(something), len(list(self.datablob.training.unbatch())))

    def test_evaluate(self):
        """Test that KerasModel.evaluate() works."""
        # Demonstrate the ability for the model to evaluate its own test set.
        on_test_set = self.keras_model.evaluate(verbose=0)
        # It is 4 because the model has 3 metrics in addition to loss.
        self.assertEqual(len(on_test_set), 4)
        # Assert that the loss is somewhat greater than 0.
        self.assertTrue(on_test_set[0] > 0.01)

        # Now demonstrate the ability to evaluate a custom dataset.
        on_provided_data = self.keras_model.evaluate(
            dataset=tf.data.Dataset.from_tensor_slices(tf.zeros(shape=(3, 5))).batch(2),
            verbose=0,
        )
        self.assertEqual(len(on_provided_data), 4)
        self.assertAlmostEqual(on_provided_data[0], 0.0)

    def test_fit_invalid_profile_batch(self):
        """Test that KerasModel.fit() needs tensorboard_directory to specify profile_batch."""
        model = sp.KerasModel(
            datablob=self.datablob,
            model_template=self.model_template,
        )
        with self.assertRaises(ValueError):
            model.fit(final_epoch=3, verbose=0, profile_batch=(1, 2))

    def test_fit_with_logging_args_default(self):
        """Test training an epoch with the default logging arguments."""
        # Make sure that no exceptions are thrown.
        self.keras_model.fit(verbose=0, final_epoch=1)

    def test_fit_with_log_batches(self):
        """Test training an epoch while logging batches to the default logger."""
        # Make sure that no exceptions are thrown.
        self.keras_model.fit(verbose=0, final_epoch=1, log_batches=True)

    def test_fit_with_log_epochs(self):
        """Test training an epoch while logging epochs to the default logger."""
        # Make sure that no exceptions are thrown.
        self.keras_model.fit(verbose=0, final_epoch=1, log_epochs=True)

    def test_fit_with_log_batches_and_log_epochs(self):
        """Test training an epoch while logging batches and epochs to the default logger."""
        # Make sure that no exceptions are thrown.
        self.keras_model.fit(
            verbose=0, final_epoch=1, log_batches=True, log_epochs=True
        )

    def test_fit_with_logging_args_default_custom_logger(self):
        """Test that we do NOT log to a custom logger if log_batches = log_epochs = False."""
        custom_logger = logging.Logger("testlogger")
        with unittest.mock.patch.object(custom_logger, "info") as mock_logger_info:
            self.keras_model.fit(verbose=0, final_epoch=1, logger=custom_logger)
            mock_logger_info.assert_not_called()

    def test_fit_with_log_batches_custom_logger(self):
        """Test training while logging batches to a custom logger."""
        custom_logger = logging.Logger("testlogger")
        with unittest.mock.patch.object(custom_logger, "info") as mock_logger_info:
            self.keras_model.fit(
                verbose=0, final_epoch=1, logger=custom_logger, log_batches=True
            )
            mock_logger_info.assert_called()

    def test_fit_with_log_epochs_custom_logger(self):
        """Test training while logging epoches to a custom logger."""
        custom_logger = logging.Logger("testlogger")
        with unittest.mock.patch.object(custom_logger, "info") as mock_logger_info:
            self.keras_model.fit(
                verbose=0, final_epoch=1, logger=custom_logger, log_epochs=True
            )
            mock_logger_info.assert_called()

    def test_fit_with_log_batches_and_log_epochs_custom_logger(self):
        """Test training while logging both batches and epochs to a custom logger."""
        custom_logger = logging.Logger("testlogger")
        with unittest.mock.patch.object(custom_logger, "info") as mock_logger_info:
            self.keras_model.fit(
                verbose=0,
                final_epoch=1,
                logger=custom_logger,
                log_batches=True,
                log_epochs=True,
            )
            mock_logger_info.assert_called()

    def test_fit_with_tensorboard(self):
        """Test that we can enable the TensorBoard callback."""
        with tempfile.TemporaryDirectory() as tensorboard_directory:
            model = sp.KerasModel(
                datablob=self.datablob,
                model_template=self.model_template,
            )
            assert_directory(tensorboard_directory, [])
            model.fit(
                final_epoch=2,
                verbose=0,
                tensorboard_directory=tensorboard_directory,
            )
            assert_directory(tensorboard_directory, [model.name])
            model_tb_dir = os.path.join(tensorboard_directory, model.name)
            assert_directory(model_tb_dir, ["train", "validation"])
            model.fit(
                final_epoch=3,
                verbose=0,
                tensorboard_directory=tensorboard_directory,
                profile_batch=(1, 2),
            )
            assert os.path.exists(os.path.join(model_tb_dir, "plugins", "profile"))

    def test_fit_with_steps_per_epoch(self):
        """Test that we can pass ``steps_per_epoch`` when fitting models."""
        datablob = MyDataBlobRepeating(hyperparams=dict(rows=10, cols=5)).batch(2)
        num_epochs = 3
        for steps_per_epoch in [1, 2, 3, 4, 5]:
            with self.subTest(steps_per_epoch=steps_per_epoch):
                model = sp.KerasModel(
                    datablob=datablob,
                    model_template=self.model_template,
                )
                last_batch = 0

                def on_batch_end(batch, logs):  # pylint: disable=unused-argument
                    nonlocal last_batch
                    last_batch += 1

                fit_kwargs = dict(
                    final_epoch=num_epochs,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.LambdaCallback(
                            on_batch_end=on_batch_end,
                        )
                    ],
                )
                model.fit(
                    steps_per_epoch=steps_per_epoch,
                    validation_steps_per_epoch=1,
                    **fit_kwargs,
                )
                self.assertEqual(last_batch, num_epochs * steps_per_epoch)

    @requires_sqlite_json
    def test_fit_with_train_store(self):
        """Test that KerasModel.fit() can log to the TrainStore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_filename = os.path.join(temp_dir, "train_store.sqlite3")
            with sp.TrainStore.from_filesystem(filename=sqlite_filename) as train_store:
                # Create and fit the model.
                # When we pass the train store, it will save the datablob,
                # model, and model template.
                model = sp.KerasModel(
                    datablob=self.datablob,
                    model_template=self.model_template,
                )
                model.fit(final_epoch=3, verbose=0, train_store=train_store)

                # Check that the models table contains the 1 model we saved.
                models_df = train_store.list_models()
                self.assertEqual(
                    sorted(models_df.columns),
                    [
                        "datablob_group_name",
                        "datablob_name",
                        "dbh__cols",
                        "dbh__rows",
                        "model_class_name",
                        "model_last_modified",
                        "model_name",
                        "model_template_group_name",
                        "model_template_name",
                        "mth__layer_1_units",
                        "mth__loss",
                        "mth__optimizer",
                    ],
                )
                self.assertEqual(
                    models_df["datablob_group_name"].tolist(),
                    [self.datablob.group_name],
                )
                self.assertEqual(
                    models_df["datablob_name"].tolist(), [self.datablob.name]
                )
                self.assertEqual(
                    models_df["dbh__cols"].tolist(), [self.datablob.hyperparams.cols]
                )
                self.assertEqual(
                    models_df["dbh__rows"].tolist(), [self.datablob.hyperparams.rows]
                )
                self.assertEqual(models_df["model_class_name"].tolist(), ["KerasModel"])
                self.assertEqual(models_df["model_name"].tolist(), [model.name])
                self.assertEqual(
                    models_df["model_template_group_name"].tolist(),
                    [self.model_template.group_name],
                )
                self.assertEqual(
                    models_df["model_template_name"].tolist(),
                    [self.model_template.name],
                )
                self.assertEqual(
                    models_df["mth__layer_1_units"].tolist(),
                    [self.model_template.hyperparams.layer_1_units],
                )
                self.assertEqual(
                    models_df["mth__loss"].tolist(),
                    [self.model_template.hyperparams.loss],
                )
                self.assertEqual(
                    models_df["mth__optimizer"].tolist(),
                    [self.model_template.hyperparams.optimizer],
                )

                # Check that the model_epochs table contains the 3 epochs we saved.
                model_epochs_df = train_store.list_model_epochs()
                self.assertEqual(
                    sorted(model_epochs_df.columns),
                    [
                        "epoch_num",
                        "last_modified",
                        "metric__binary_accuracy",
                        "metric__loss",
                        "metric__precision",
                        "metric__recall",
                        "metric__val_binary_accuracy",
                        "metric__val_loss",
                        "metric__val_precision",
                        "metric__val_recall",
                        "model_name",
                        "steps_per_epoch",
                        "validation_steps_per_epoch",
                    ],
                )
                self.assertEqual(model_epochs_df["epoch_num"].tolist(), [1, 2, 3])
                self.assertEqual(
                    model_epochs_df["model_name"].tolist(),
                    [model.name, model.name, model.name],
                )
                self.assertEqual(
                    model_epochs_df["steps_per_epoch"].tolist(), [None, None, None]
                )
                self.assertEqual(
                    model_epochs_df["validation_steps_per_epoch"].tolist(),
                    [None, None, None],
                )

    @requires_sqlite_json
    def test_fit_with_train_store_with_steps_per_epoch(self):
        """Test that KerasModel.fit() can log to the TrainStore including steps_per_epoch."""
        for steps_per_epoch in [1, 2, 3]:
            for validation_steps_per_epoch in [1, 2, 3]:
                with self.subTest(
                    steps_per_epoch=steps_per_epoch,
                    validation_steps_per_epoch=validation_steps_per_epoch,
                ):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        sqlite_filename = os.path.join(temp_dir, "train_store.sqlite3")
                        with sp.TrainStore.from_filesystem(
                            filename=sqlite_filename
                        ) as train_store:
                            # Create and fit the model.
                            # When we pass the train store, it will save the datablob,
                            # model, and model template.
                            model = sp.KerasModel(
                                datablob=self.datablob.repeat(),
                                model_template=self.model_template,
                            )
                            model.fit(
                                final_epoch=3,
                                verbose=0,
                                train_store=train_store,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps_per_epoch=validation_steps_per_epoch,
                            )

                            # Check that the models table contains the 1 model we saved.
                            models_df = train_store.list_models()
                            self.assertEqual(
                                sorted(models_df.columns),
                                [
                                    "datablob_group_name",
                                    "datablob_name",
                                    "dbh__cols",
                                    "dbh__rows",
                                    "model_class_name",
                                    "model_last_modified",
                                    "model_name",
                                    "model_template_group_name",
                                    "model_template_name",
                                    "mth__layer_1_units",
                                    "mth__loss",
                                    "mth__optimizer",
                                ],
                            )
                            self.assertEqual(
                                models_df["datablob_group_name"].tolist(),
                                [self.datablob.group_name],
                            )
                            self.assertEqual(
                                models_df["datablob_name"].tolist(),
                                [self.datablob.name],
                            )
                            self.assertEqual(
                                models_df["dbh__cols"].tolist(),
                                [self.datablob.hyperparams.cols],
                            )
                            self.assertEqual(
                                models_df["dbh__rows"].tolist(),
                                [self.datablob.hyperparams.rows],
                            )
                            self.assertEqual(
                                models_df["model_class_name"].tolist(), ["KerasModel"]
                            )
                            self.assertEqual(
                                models_df["model_name"].tolist(), [model.name]
                            )
                            self.assertEqual(
                                models_df["model_template_group_name"].tolist(),
                                [self.model_template.group_name],
                            )
                            self.assertEqual(
                                models_df["model_template_name"].tolist(),
                                [self.model_template.name],
                            )
                            self.assertEqual(
                                models_df["mth__layer_1_units"].tolist(),
                                [self.model_template.hyperparams.layer_1_units],
                            )
                            self.assertEqual(
                                models_df["mth__loss"].tolist(),
                                [self.model_template.hyperparams.loss],
                            )
                            self.assertEqual(
                                models_df["mth__optimizer"].tolist(),
                                [self.model_template.hyperparams.optimizer],
                            )

                            # Check that the model_epochs table contains the 3 epochs we saved.
                            model_epochs_df = train_store.list_model_epochs()
                            self.assertEqual(
                                sorted(model_epochs_df.columns),
                                [
                                    "epoch_num",
                                    "last_modified",
                                    "metric__binary_accuracy",
                                    "metric__loss",
                                    "metric__precision",
                                    "metric__recall",
                                    "metric__val_binary_accuracy",
                                    "metric__val_loss",
                                    "metric__val_precision",
                                    "metric__val_recall",
                                    "model_name",
                                    "steps_per_epoch",
                                    "validation_steps_per_epoch",
                                ],
                            )
                            self.assertEqual(
                                model_epochs_df["epoch_num"].tolist(), [1, 2, 3]
                            )
                            self.assertEqual(
                                model_epochs_df["model_name"].tolist(),
                                [model.name, model.name, model.name],
                            )
                            self.assertEqual(
                                model_epochs_df["steps_per_epoch"].tolist(),
                                [steps_per_epoch, steps_per_epoch, steps_per_epoch],
                            )
                            self.assertEqual(
                                model_epochs_df["validation_steps_per_epoch"].tolist(),
                                [
                                    validation_steps_per_epoch,
                                    validation_steps_per_epoch,
                                    validation_steps_per_epoch,
                                ],
                            )
