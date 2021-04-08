Organize your machine learning experiments with ScalarStop
==========================================================

ScalarStop is a framework written in Python that helps you keep track of datasets, models, hyperparameters, and training metrics in machine learning experiments.

Installation
------------

ScalarStop is `available on PyPI <https://pypi.org/project/scalarstop/>`_.
You can install by running the command:

.. code:: bash

    pip3 install scalarstop


Usage
-----

Read the `ScalarStop Tutorial <https://github.com/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_ to learn the core concepts behind ScalarStop and how to structure your datasets and models.

Afterwards, you might want to dig deeper into the `ScalarStop Documentation <https://docs.scalarstop.com>`_. In general, a typical ScalarStop workflow involves four steps:

1. Organize your datasets with `scalarstop.datablob <https://www.scalarstop.com/en/latest/autoapi/scalarstop/datablob/#module-scalarstop.datablob>`_.
2. Describe your machine learning model architectures using `scalarstop.model_template <https://www.scalarstop.com/en/latest/autoapi/scalarstop/model_template/#module-scalarstop.model_template>`_.
3. Load, train, and save machine learning models with `scalarstop.model <https://www.scalarstop.com/en/latest/autoapi/scalarstop/model/#module-scalarstop.model>`_.
4. Save hyperparameters and training metrics to a SQLite or PostgreSQL database using `scalarstop.train_store <https://www.scalarstop.com/en/latest/autoapi/scalarstop/train_store/#module-scalarstop.train_store>`_.
