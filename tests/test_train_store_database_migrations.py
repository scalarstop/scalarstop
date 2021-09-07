"""
Test our ability to migrate to newer versions of the
TrainStore database schema.
"""
import os
import shutil
import tempfile
import unittest

from sqlalchemy import inspect

import scalarstop as sp


def get_column_names(inspector, table_name):
    """
    Return the actual columns that are in the database table,
    and not just the columns that our SQLAlchemy model expects.
    """
    return sorted([col["name"] for col in inspector.get_columns(table_name)])


class TestTrainStoreMigrations(unittest.TestCase):
    """Tests for the TrainStore's automatic migration features."""

    @classmethod
    def setUpClass(cls):
        database_directory = os.path.join(
            os.path.dirname(__file__), "database_fixtures"
        )
        cls.database_filenames = [
            os.path.join(database_directory, filename)
            for filename in os.listdir(database_directory)
            if filename.endswith(".sqlite3")
        ]

    def test_migration(self):
        """Test that we have all of the expected columns after a migration."""
        # We copy each SQLite3 fixture to a temporary directory because
        # the database migration process mutates each file.
        with tempfile.TemporaryDirectory() as tempdir:
            for filename in self.database_filenames:
                basename = os.path.basename(filename)
                with self.subTest(os.path.basename(filename)):
                    temp_filename = os.path.join(tempdir, basename)
                    shutil.copy2(filename, temp_filename)
                    with sp.TrainStore.from_filesystem(
                        filename=temp_filename
                    ) as train_store:
                        inspector = inspect(train_store.connection)
                        # DataBlob
                        datablob_cols = get_column_names(
                            inspector, "scalarstop__datablob"
                        )
                        self.assertEqual(
                            datablob_cols,
                            [
                                "datablob_group_name",
                                "datablob_hyperparams",
                                "datablob_last_modified",
                                "datablob_name",
                            ],
                        )
                        # ModelTemplate
                        mt_cols = get_column_names(
                            inspector, "scalarstop__model_template"
                        )
                        self.assertEqual(
                            mt_cols,
                            [
                                "model_template_group_name",
                                "model_template_hyperparams",
                                "model_template_last_modified",
                                "model_template_name",
                            ],
                        )
                        # Model
                        model_cols = get_column_names(inspector, "scalarstop__model")
                        self.assertEqual(
                            model_cols,
                            [
                                "datablob_name",
                                "model_class_name",
                                "model_last_modified",
                                "model_name",
                                "model_template_name",
                            ],
                        )
                        # ModelEpoch
                        model_epoch_cols = get_column_names(
                            inspector, "scalarstop__model_epoch"
                        )
                        self.assertEqual(
                            model_epoch_cols,
                            [
                                "model_epoch_id",
                                "model_epoch_last_modified",
                                "model_epoch_metrics",
                                "model_epoch_num",
                                "model_name",
                            ],
                        )
