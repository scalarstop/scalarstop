
Organize your machine learning experiments with ScalarStop
==========================================================

ScalarStop helps you train machine learning models by:

* creating a system to uniquely name datasets, model
  architectures, trained models, and their
  hyperparameters.
* saving and loading datasets and models to/from the
  filesystem in a consistent way.
* recording dataset and model names, hyperparameters, and
  training metrics to a SQLite or PostgreSQL database.

Installation
------------

ScalarStop is available on PyPI. You can install it from
the command line using::

    pip3 install scalarstop

Usage
-----
The best way to learn ScalarStop is to follow the
`Official ScalarStop Tutorial <https://nbviewer.jupyter.org/github/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_.

Afterwards, you might want to dig deeper into the ScalarStop documentation.
In general, a typical ScalarStop workflow involves four steps:

1. Organize your datasets with :py:mod:`scalarstop.datablob`.
2. Describe your machine learning model architectures using :py:mod:`scalarstop.model_template`.
3. Load, train, and save machine learning models with :py:mod:`scalarstop.model`.
4. Save hyperparameters and training metrics to a SQLite or PostgreSQL database using :py:mod:`scalarstop.train_store`.
