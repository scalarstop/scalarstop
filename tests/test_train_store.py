"""
Unit tests for scalarstop.train_store.
"""
import os
import random
import tempfile
import unittest

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import scalarstop as sp
from tests.fixtures import (
    MyDataBlob,
    MyModelTemplate,
    requires_external_database,
    requires_sqlite_json,
)


class TrainStoreUnits:  # pylint: disable=no-member
    """
    Tests for :py:class:`~scalarstop.TrainStore`.

    This is a parent class that contains train store tests,
    but does not set up a specific TrainStore instance. We leave
    that to the subclasses, allowing us to run the same tests
    against multiple database backends.
    """

    def setUp(self):
        """Setup."""
        self._models_directory_context = tempfile.TemporaryDirectory()
        self.models_directory = self._models_directory_context.name

        self.datablob = MyDataBlob(hyperparams=dict(rows=10, cols=5)).batch(2)
        self.model_template = MyModelTemplate(hyperparams=dict(layer_1_units=2))
        self.model = sp.KerasModel(
            datablob=self.datablob,
            model_template=self.model_template,
        )

    def tearDown(self):
        """Teardown."""
        self._models_directory_context.cleanup()

    def test_insert_datablob(self):
        """Test :py:meth:`~scalarstop.TrainStore.insert_datablob`."""
        self.train_store.insert_datablob(self.datablob)
        self.assertEqual(len(self.train_store.list_datablobs()), 1)

        # Assert that we raise an exception when inserting the same DataBlob.
        with self.assertRaises(IntegrityError):
            self.train_store.insert_datablob(self.datablob)
        with self.assertRaises(IntegrityError):
            self.train_store.insert_datablob_by_str(
                name=self.datablob.name, group_name="", hyperparams=None
            )

        # Assert that we can suppress that exception.
        self.train_store.insert_datablob(self.datablob, ignore_existing=True)

        # Examine what we just added.
        df = self.train_store.list_datablobs()
        self.assertEqual(len(df), 1)
        self.assertEqual(
            sorted(df.keys()),
            [
                "group_name",
                "hyperparams",
                "last_modified",
                "name",
            ],
        )
        self.assertEqual(df["name"].tolist(), ["MyDataBlob-mftoseayyazof6cibziqosm"])
        self.assertEqual(df["group_name"].tolist(), ["MyDataBlob"])
        self.assertEqual(df["hyperparams"].tolist(), [dict(rows=10, cols=5)])

    def test_insert_model_template(self):
        """Test :py:meth:`~scalarstop.TrainStore.insert_model_template`."""
        self.train_store.insert_model_template(self.model_template)
        self.assertEqual(len(self.train_store.list_model_templates()), 1)

        # Assert that we raise an exception when inserting the same ModelTemplate.
        with self.assertRaises(IntegrityError):
            self.train_store.insert_model_template(self.model_template)
        with self.assertRaises(IntegrityError):
            self.train_store.insert_model_template_by_str(
                name=self.model_template.name,
                group_name="",
                hyperparams=None,
            )

        # Assert that we can suppress that exception.
        self.train_store.insert_model_template(
            self.model_template, ignore_existing=True
        )

        # Examine what we just added.
        df = self.train_store.list_model_templates()
        self.assertEqual(len(df), 1)
        self.assertEqual(
            sorted(df.keys()),
            [
                "group_name",
                "hyperparams",
                "last_modified",
                "name",
            ],
        )
        self.assertEqual(
            df["name"].tolist(),
            ["MyModelTemplate-29utnha73paz6fvwivrs5fn6"],
        )
        self.assertEqual(df["group_name"].tolist(), ["MyModelTemplate"])
        self.assertEqual(
            df["hyperparams"].tolist(),
            [dict(layer_1_units=2, loss="binary_crossentropy", optimizer="adam")],
        )

    def test_insert_model(self):
        """Test :py:meth:`~scalarstop.TrainStore.insert_model`."""
        self.train_store.insert_datablob(self.datablob)
        self.train_store.insert_model_template(self.model_template)
        self.train_store.insert_model(self.model)
        self.assertEqual(len(self.train_store.list_models()), 1)

        # Assert that we raise an exception when inserting another with the same name.
        with self.assertRaises(IntegrityError):
            self.train_store.insert_model(self.model)
        with self.assertRaises(IntegrityError):
            self.train_store.insert_model_by_str(
                name=self.model.name,
                model_class_name="",
                datablob_name="",
                model_template_name="",
            )

        # Assert that we can suppress that exception.
        self.train_store.insert_model(self.model, ignore_existing=True)

        # Examine what we just added.
        df = self.train_store.list_models()
        self.assertEqual(len(df), 1)
        self.assertEqual(
            sorted(df.keys()),
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
        self.assertEqual(df["model_class_name"].tolist(), ["KerasModel"])
        self.assertEqual(
            df["datablob_name"].tolist(), ["MyDataBlob-mftoseayyazof6cibziqosm"]
        )
        self.assertEqual(df["datablob_group_name"].tolist(), ["MyDataBlob"])
        self.assertEqual(
            df["model_template_name"].tolist(),
            ["MyModelTemplate-29utnha73paz6fvwivrs5fn6"],
        )
        self.assertEqual(df["model_template_group_name"].tolist(), ["MyModelTemplate"])
        self.assertEqual(
            df["model_name"].tolist(),
            [
                "mt_MyModelTemplate-29utnha73paz6fvwivrs5fn6__d_MyDataBlob-mftoseayyazof6cibziqosm"
            ],
        )

    def test_insert_model_epoch(self):
        """Test :py:meth:`~scalarstop.TrainStore.insert_model_epoch`."""
        # Insert our first ModelEpoch.
        self.train_store.insert_datablob(self.datablob)
        self.train_store.insert_model_template(self.model_template)
        self.train_store.insert_model(self.model)
        self.train_store.insert_model_epoch(
            model_name=self.model.name, epoch_num=0, metrics=dict(loss=3, accuracy=5)
        )
        self.assertEqual(len(self.train_store.list_model_epochs()), 1)

        # Assert that we raise an exception when inserting another
        # with the same name and epoch number.
        with self.assertRaises(IntegrityError):
            self.train_store.insert_model_epoch(
                model_name=self.model.name, epoch_num=0, metrics=dict()
            )

        # Assert that we can suppress that exception.
        self.train_store.insert_model_epoch(
            model_name=self.model.name,
            epoch_num=0,
            metrics=dict(loss=3, accuracy=5),
            ignore_existing=True,
        )

        # Examine what we inserted.
        df = self.train_store.list_model_epochs()
        self.assertEqual(len(df), 1)
        self.assertEqual(
            sorted(df.keys()),
            [
                "epoch_num",
                "last_modified",
                "metric__accuracy",
                "metric__loss",
                "model_name",
            ],
        )
        self.assertEqual(df["metric__accuracy"].tolist(), [5])
        self.assertEqual(df["metric__loss"].tolist(), [3])
        self.assertEqual(df["epoch_num"].tolist(), [0])
        self.assertEqual(df["model_name"].tolist(), [self.model.name])

    def test_bulk_insert_model_epochs(self):
        """Test :py:meth:`~scalarstop.TrainStore.bulk_insert_model_epochs`."""
        # Set everything up.
        self.train_store.insert_datablob(self.datablob)
        self.train_store.insert_model_template(self.model_template)
        self.train_store.insert_model(self.model)

        # Train 3 epochs and do not log them into the TrainStore.
        self.model.fit(final_epoch=3, verbose=0)

        # Train 3 more epochs and DO log these into the TrainStore.
        self.model.fit(final_epoch=6, verbose=0, train_store=self.train_store)

        # Assert that the TrainStore only has the epochs that we added to it.
        model_epochs_1 = self.train_store.list_model_epochs(self.model.name)
        self.assertEqual(len(model_epochs_1), 3)
        self.assertEqual(model_epochs_1["epoch_num"].tolist(), [4, 5, 6])

        # Train 3 more epochs without including them in the TrainStore.
        self.model.fit(final_epoch=9, verbose=0)

        # Assert that the TrainStore only has the epochs that we added to it.
        model_epochs_2 = self.train_store.list_model_epochs(self.model.name)
        self.assertEqual(len(model_epochs_2), 3)
        self.assertEqual(model_epochs_2["epoch_num"].tolist(), [4, 5, 6])

        # Now we use the bulk insert to add all of the epochs that we forgot to log.
        self.train_store.bulk_insert_model_epochs(self.model)
        model_epochs_3 = self.train_store.list_model_epochs(self.model.name)
        self.assertEqual(len(model_epochs_3), 9)
        self.assertEqual(
            model_epochs_3["epoch_num"].tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )

    def test_get_current_epoch(self):
        """Test :py:meth:`~scalarstop.TrainStore.get_current_epoch`."""
        self.train_store.insert_datablob(self.datablob)

        mt1 = MyModelTemplate(hyperparams=dict(layer_1_units=2))
        self.train_store.insert_model_template(mt1)

        mt2 = MyModelTemplate(hyperparams=dict(layer_1_units=3))
        self.train_store.insert_model_template(mt2)

        mt3 = MyModelTemplate(hyperparams=dict(layer_1_units=4))
        self.train_store.insert_model_template(mt3)

        model1 = sp.KerasModel(
            datablob=self.datablob,
            model_template=mt1,
        )
        self.train_store.insert_model(model1)

        model2 = sp.KerasModel(
            datablob=self.datablob,
            model_template=mt2,
        )
        self.train_store.insert_model(model2)

        model3 = sp.KerasModel(
            datablob=self.datablob,
            model_template=mt3,
        )
        self.train_store.insert_model(model3)
        self.train_store.insert_model_epoch(
            model_name=model1.name, epoch_num=0, metrics=dict(loss=3, accuracy=5)
        )
        self.train_store.insert_model_epoch(
            model_name=model1.name, epoch_num=1, metrics=dict(loss=3, accuracy=5)
        )
        self.train_store.insert_model_epoch(
            model_name=model2.name, epoch_num=1, metrics=dict(loss=3, accuracy=5)
        )
        self.train_store.insert_model_epoch(
            model_name=model2.name, epoch_num=2, metrics=dict(loss=3, accuracy=5)
        )
        self.train_store.insert_model_epoch(
            model_name=model3.name, epoch_num=500, metrics=dict(loss=3, accuracy=5)
        )

        gce = self.train_store.get_current_epoch
        self.assertEqual(gce(model1.name), 1)
        self.assertEqual(gce(model2.name), 2)
        self.assertEqual(gce(model3.name), 500)
        self.assertEqual(gce("nonexistent"), 0)

    def test_get_best_model(self):
        """Test :py:meth:`~scalarstop.TrainStore.get_best_model`."""
        datablobs = [
            MyDataBlob(hyperparams=dict(rows=5, cols=5)).batch(2),
            MyDataBlob(hyperparams=dict(rows=7, cols=5)).batch(2),
            MyDataBlob(hyperparams=dict(rows=9, cols=5)).batch(2),
        ]
        for db in datablobs:
            self.train_store.insert_datablob(db)

        model_templates = [
            MyModelTemplate(hyperparams=dict(layer_1_units=2)),
            MyModelTemplate(hyperparams=dict(layer_1_units=3)),
            MyModelTemplate(hyperparams=dict(layer_1_units=4)),
        ]
        for mt in model_templates:
            self.train_store.insert_model_template(mt)

        random.shuffle(datablobs)
        random.shuffle(model_templates)
        for db in datablobs:
            for mt in model_templates:
                model = sp.KerasModel(
                    datablob=db,
                    model_template=mt,
                )
                self.train_store.insert_model(model)
                for epoch_num in range(3):
                    metrics = dict(
                        my_metric=db.hyperparams.rows
                        * mt.hyperparams.layer_1_units
                        * epoch_num
                    )
                    self.train_store.insert_model_epoch(
                        model_name=model.name,
                        epoch_num=epoch_num,
                        metrics=metrics,
                    )
                    if metrics["my_metric"] == 72:
                        expected_best = model

        actual_best = self.train_store.get_best_model(
            metric_name="my_metric",
            metric_direction="max",
            datablob_group_name="MyDataBlob",
            model_template_group_name="MyModelTemplate",
        )
        self.assertEqual(
            sorted(sp.dataclasses.asdict(actual_best)),
            [
                "datablob_group_name",
                "datablob_hyperparams",
                "datablob_name",
                "model_class_name",
                "model_epoch_metrics",
                "model_last_modified",
                "model_name",
                "model_template_group_name",
                "model_template_hyperparams",
                "model_template_name",
                "sort_metric_name",
                "sort_metric_value",
            ],
        )
        self.assertEqual(
            actual_best.datablob_group_name, expected_best.datablob.group_name
        )
        self.assertEqual(
            actual_best.datablob_hyperparams,
            sp.dataclasses.asdict(expected_best.datablob.hyperparams),
        )
        self.assertEqual(actual_best.datablob_name, expected_best.datablob.name)
        self.assertEqual(actual_best.model_class_name, "KerasModel")
        self.assertEqual(actual_best.model_epoch_metrics, dict(my_metric=72))
        self.assertEqual(actual_best.model_name, expected_best.name)
        self.assertEqual(
            actual_best.model_template_group_name,
            expected_best.model_template.group_name,
        )
        self.assertEqual(
            actual_best.model_template_hyperparams,
            sp.dataclasses.asdict(expected_best.model_template.hyperparams),
        )
        self.assertEqual(
            actual_best.model_template_name, expected_best.model_template.name
        )
        self.assertEqual(actual_best.sort_metric_name, "my_metric")
        self.assertEqual(actual_best.sort_metric_value, 72)


@requires_external_database
class TestTrainStoreWithExternalDatabase(TrainStoreUnits, unittest.TestCase):
    """
    Runs TrainStore unit test against an external non-SQLite database.

    To run these tests, provide a valid SQLAlchemy database connection
    string in the environment variable `TRAIN_STORE_CONNECTION_STRING`.
    """

    def setUp(self):
        super().setUp()
        self.connection_string = os.environ["TRAIN_STORE_CONNECTION_STRING"]
        # Every time we run a unit test, we should connect to the database
        # and drop the database tables.
        with sp.TrainStore(connection_string=self.connection_string) as train_store:
            with train_store.connection.begin():
                train_store.table.metadata.drop_all(train_store.connection)
        self.train_store = sp.TrainStore(connection_string=self.connection_string)

    def tearDown(self):
        self.train_store.close()
        super().tearDown()

    def test_postgres_multiple_table_name_prefixes(self):
        """Test multiple table name prefixes with the PostgreSQL database."""
        with sp.TrainStore(
            connection_string=self.connection_string, table_name_prefix="prefix_2__"
        ) as train_store_2:
            with sp.TrainStore(
                connection_string=self.connection_string, table_name_prefix="prefix_3__"
            ) as train_store_3:
                for train_store in (self.train_store, train_store_2, train_store_3):
                    with train_store.connection.begin():
                        tables = {
                            row[0]
                            for row in train_store.connection.execute(
                                text("SELECT tablename FROM pg_catalog.pg_tables")
                            ).fetchall()
                        }
                    self.assertIn("scalarstop__datablob", tables)
                    self.assertIn("scalarstop__model", tables)
                    self.assertIn("scalarstop__model_epoch", tables)
                    self.assertIn("scalarstop__model_template", tables)

                    self.assertIn("prefix_2__datablob", tables)
                    self.assertIn("prefix_2__model", tables)
                    self.assertIn("prefix_2__model_epoch", tables)
                    self.assertIn("prefix_2__model_template", tables)

                    self.assertIn("prefix_3__datablob", tables)
                    self.assertIn("prefix_3__model", tables)
                    self.assertIn("prefix_3__model_epoch", tables)
                    self.assertIn("prefix_3__model_template", tables)


@requires_sqlite_json
class TestTrainStoreWithSQLite(TrainStoreUnits, unittest.TestCase):
    """Runs TrainStore unit tests against a SQLite backend."""

    def setUp(self):
        super().setUp()
        self._sqlite_directory_context = tempfile.TemporaryDirectory()
        self.sqlite_filename = os.path.join(
            self._sqlite_directory_context.name, "train_store.sqlite3"
        )
        self.train_store = sp.TrainStore.from_filesystem(
            filename=self.sqlite_filename,
        )

    def tearDown(self):
        super().tearDown()
        self.train_store.close()
        self._sqlite_directory_context.cleanup()

    def test_sqlite_multiple_table_name_prefixes(self):
        """Test multiple table name prefixes with the SQLite3 database."""
        with sp.TrainStore.from_filesystem(
            filename=self.sqlite_filename, table_name_prefix="prefix_2__"
        ) as train_store_2:
            with sp.TrainStore.from_filesystem(
                filename=self.sqlite_filename, table_name_prefix="prefix_3__"
            ) as train_store_3:
                for train_store in (self.train_store, train_store_2, train_store_3):
                    with train_store.connection.begin():
                        tables = {
                            row[0]
                            for row in train_store.connection.execute(
                                text(
                                    "SELECT name FROM sqlite_master WHERE type='table'"
                                )
                            ).fetchall()
                        }
                    self.assertIn("scalarstop__datablob", tables)
                    self.assertIn("scalarstop__model", tables)
                    self.assertIn("scalarstop__model_epoch", tables)
                    self.assertIn("scalarstop__model_template", tables)

                    self.assertIn("prefix_2__datablob", tables)
                    self.assertIn("prefix_2__model", tables)
                    self.assertIn("prefix_2__model_epoch", tables)
                    self.assertIn("prefix_2__model_template", tables)

                    self.assertIn("prefix_3__datablob", tables)
                    self.assertIn("prefix_3__model", tables)
                    self.assertIn("prefix_3__model_epoch", tables)
                    self.assertIn("prefix_3__model_template", tables)
