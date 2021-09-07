#!/usr/bin/env python3
import argparse
import os
import subprocess

import scalarstop as sp
from tests.fixtures import MyDataBlob, MyModelTemplate


def create_database(filename):
    with sp.TrainStore.from_filesystem(filename=filename) as train_store:
        datablobs = [
            MyDataBlob(hyperparams=dict(rows=5, cols=5)).batch(2),
            MyDataBlob(hyperparams=dict(rows=7, cols=5)).batch(2),
            MyDataBlob(hyperparams=dict(rows=9, cols=5)).batch(2),
        ]
        for db in datablobs:
            train_store.insert_datablob(db, ignore_existing=True)

        model_templates = [
            MyModelTemplate(hyperparams=dict(layer_1_units=2)),
            MyModelTemplate(hyperparams=dict(layer_1_units=3)),
            MyModelTemplate(hyperparams=dict(layer_1_units=4)),
        ]
        for mt in model_templates:
            train_store.insert_model_template(mt, ignore_existing=True)
        for db in datablobs:
            for mt in model_templates:
                model = sp.KerasModel(datablob=db, model_template=mt)
                model.fit(final_epoch=3, verbose=0, train_store=train_store)


def main():
    parser = argparse.ArgumentParser(
        description="Create a dummy SQLite3 database to "
        "use as a test fixture in TrainStore unit tests."
    )
    parser.add_argument(
        "--sqlite3-filename",
        required=True,
        type=str,
        help="The filename to write the SQLite3 database to.",
    )
    args = parser.parse_args()
    create_database(args.sqlite3_filename)


def main2():
    latest_git_tag = subprocess.run(
        ["git", "describe", "--tags"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    filename = os.path.join(
        os.path.dirname(__file__),
        "tests",
        "database_fixtures",
        latest_git_tag + ".sqlite3",
    )
    create_database(filename)


if __name__ == "__main__":
    main2()
