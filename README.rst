.. image:: https://www.scalarstop.com/en/latest/_static/1x/logo-color-black-on-white-1x.png

Keep track of your machine learning experiments with ScalarStop.
================================================================

ScalarStop is a Python framework for reproducible machine learning research.

It was written and open-sourced at `Neocrym <https://www.neocrym.com>`_, where it is used to train thousands of models every week.

ScalarStop can help you:

* organize datasets and models with *content-addressable* names.
* save/load datasets and models to/from the filesystem.
* record hyperparameters and metrics to a relational database.

System requirements
-------------------
ScalarStop is a Python package that requires Python 3.8 or newer.

Currently, ScalarStop only supports tracking
`tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
datasets and `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_
models. As such, ScalarStop requires TensorFlow 2.3.0 or newer.

We encourage anybody that would like to add support for other
machine learning frameworks to ScalarStop. :)

Installation
------------

ScalarStop is `available on PyPI <https://pypi.org/project/scalarstop/>`_.
You can install by running the command:

.. code:: bash

    python3 -m pip install scalarstop

If you would like to make changes to ScalarStop, you can `clone the repository <https://github.com/scalarstop/scalarstop>`_
from GitHub.

.. code:: bash

    git clone https://github.com/scalarstop/scalarstop.git
    cd scalarstop
    python3 -m pip install .

Usage
-----

Read the `ScalarStop Tutorial <https://github.com/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_ to learn the core concepts behind ScalarStop and how to structure your datasets and models.

Afterwards, you might want to dig deeper into the `ScalarStop Documentation <https://docs.scalarstop.com>`_. In general, a typical ScalarStop workflow involves four steps:

1. Organize your datasets with `scalarstop.datablob <https://www.scalarstop.com/en/latest/autoapi/scalarstop/datablob/#module-scalarstop.datablob>`_.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2. Describe your machine learning model architectures using `scalarstop.model_template <https://www.scalarstop.com/en/latest/autoapi/scalarstop/model_template/#module-scalarstop.model_template>`_.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3. Load, train, and save machine learning models with `scalarstop.model <https://www.scalarstop.com/en/latest/autoapi/scalarstop/model/#module-scalarstop.model>`_.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
4. Save hyperparameters and training metrics to a SQLite or PostgreSQL database using `scalarstop.train_store <https://www.scalarstop.com/en/latest/autoapi/scalarstop/train_store/#module-scalarstop.train_store>`_.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contributing to ScalarStop
--------------------------

We warmly welcome contributions to ScalarStop. Here are the technical details for getting started with adding code to ScalarStop.

Getting started
^^^^^^^^^^^^^^^
First, clone this repository from GitHub. All development happens on the ``main`` branch.

.. code:: bash

    git clone https://github.com/scalarstop/scalarstop.git

Then, run ``make install`` to install Python dependencies in a Poetry virtualenv.

You can run ``make help`` to see the other commands that are available.

Checking your code
^^^^^^^^^^^^^^^^^^
Run ``make fmt`` to automatically format code.

Run ``make lint`` to run Pylint and MyPy to check for errors.

Generating documentation
^^^^^^^^^^^^^^^^^^^^^^^^
Documentation is important! Here is how to add to it.

Generating Sphinx documentation
"""""""""""""""""""""""""""""""

You can generate a local copy of our Sphinx documentation at `scalarstop.com <https://www.scalarstop.com/en/latest/>`_ with ``make docs``.

The generated documentation can be found at ``docs/_build/dirhtml``. To view it, you should start an HTTP server in this directory, such as:

.. code:: bash

    make docs
    cd docs/_build/dirhtml
    python3 -m http.server 5000

Then visit http://localhost:5000 in your browser to preview changes to the documentation.

If you want to use Sphinx's ability to automatically generate hyperlinks to the Sphinx documentation of other Python projects, then you should configure `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ settings at the path docs/conf.py. If you need to download an objects.inv file, make sure to update the ``make update-sphinx`` command in the `Makefile <https://github.com/scalarstop/scalarstop/blob/main/Makefile>`_.

Editing the tutorial notebook
"""""""""""""""""""""""""""""
The main ScalarStop tutorial is `in a Jupyter notebook <https://github.com/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_. If you have made changes to ScalarStop, you should rerun the Jupyter notebook on your machine with your changes to make sure that it still runs without error.

Running unit tests
^^^^^^^^^^^^^^^^^^
Run ``make test`` to run all unit tests.

If you want to run a specific unit test, try running ``python3 -m poetry run python -m unittest -k {name of your test}``.

Unit tests with SQLite3
"""""""""""""""""""""""
If you are running tests using a Python interpreter that does not have the `SQLite3 JSON1 extension <https://www.sqlite.org/json1.html>`_, then `TrainStore <https://www.scalarstop.com/en/latest/autoapi/scalarstop/train_store/#module-scalarstop.train_store>`_ unit tests involving SQLite3 will be skipped. This is likely to happen if you are using Python 3.8 on Windows. If you suspect that you are missing the SQLite3 JSON1 extension, the `Django documentation has some suggestions <https://code.djangoproject.com/wiki/JSON1Extension>`_ for how to fix it.

Unit tests with PostgreSQL
""""""""""""""""""""""""""
By default, tests involving PostgreSQL are skipped. To enable PostgreSQL, run ``make test`` in a shell where the environment variable ``TRAIN_STORE_CONNECTION_STRING`` is set to a `SQLAlchemy database connection URL <https://docs.sqlalchemy.org/en/14/core/engines.html>`_--which looks something like ``"postgresql://scalarstop:changeme@localhost:5432/train_store"``. The connection URL should point to a working PostgreSQL database with an existing database and user.

The docker-compose.yml file in the root of this directory can set up a PostgreSQL instance on your local machine. If you have `Docker <https://docs.docker.com/get-docker/>`_ and `Docker Compose <https://docs.docker.com/compose/install/>`_ installed, you can start the PostgreSQL database by running ``docker-compose up`` in the same directory as the docker-compose.yml file.

Measuring test coverage
"""""""""""""""""""""""
You can run ``make test-with-coverage`` to collect Python line and branch coverage information. Afterwards, run ``make coverage-html`` to generate an HTML report of unit test coverage. You can view the report in a web browser at the path ``htmlcov/index.html``.

Credits
-------
ScalarStop's documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_ using `@pradyunsg <https://pradyunsg.me>`_'s `Furo <https://github.com/pradyunsg/furo>`_ theme and is hosted by `Read the Docs <https://readthedocs.org/>`_.
