:hide-toc:

.. toctree::
    :hidden:
    :maxdepth: 2

    Home <self>
    API Documentation <autoapi/scalarstop/index>

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: External Links

    Offical Tutorial <https://nbviewer.jupyter.org/github/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>
    ScalarStop on GitHub <https://github.com/scalarstop/scalarstop>
    ScalarStop on PyPI <https://pypi.org/project/scalarstop>

.. raw:: html

    <div class="mega-hide-on-mobile"><img src="_static/images/code-screenshot.svg" class="mega-hero-img" /></div>

    <h1 class="mega-header centered">Keep track of your machine learning experiments.</h1>

    <h2 class="mega-header centered">ScalarStop is an open-source framework for reproducible machine learning research.</h2>

    <h3 class="mega-header">ScalarStop was written and open-sourced at <a href="https://www.neocrym.com">Neocrym</a>, where it is used to train thousands of models every week.

    <h2 class="mega-header">ScalarStop can help you:</h2>

    <h3 class="mega-header scalarstop-bigbullet">organize datasets and models with <em>content-addressable</em> names.</h3>
    ScalarStop datasets and models are given automatically-generated
    names based on their hyperparameters--making them easy to version and easy to find.

    <h3 class="mega-header mega-bigbullet">save/load datasets and models to/from the filesystem.</h3>
    ScalarStop wraps existing dataset and model saving logic in TensorFlow
    for safety, correctness, and completion.

    <h3 class="mega-header mega-bigbullet">record hyperparameters and metrics to a relational database.</h3>
    ScalarStop saves dataset and model names, hyperparameters, and training
    metrics to a SQLite or PostgreSQL database.

Getting started
---------------

System requirements
^^^^^^^^^^^^^^^^^^^
ScalarStop is a Python package that requires Python 3.8 or newer.

Currently, ScalarStop only supports tracking :py:class:`tf.data.Dataset`
datasets and :py:class:`tf.keras.Model` models. As such, ScalarStop
requires TensorFlow 2.5.0 or newer.

We encourage anybody that would like to add support for other
machine learning frameworks to ScalarStop. :)

Installation
^^^^^^^^^^^^

ScalarStop is `available on PyPI <https://pypi.org/project/scalarstop/>`_.

If you are using TensorFlow on a CPU, you can install ScalarStop with the command:

.. code:: bash

    python3 -m pip install scalarstop[tensorflow]

If you are using TensorFlow with GPUs, you can install ScalarStop with the command:

.. code:: bash

    python3 -m pip install scalarstop[tensorflow-gpu]

Development
-----------

If you would like to make changes to ScalarStop, you can `clone the repository <https://github.com/scalarstop/scalarstop>`_
from GitHub.

.. code:: bash

    git clone https://github.com/scalarstop/scalarstop.git
    cd scalarstop
    python3 -m pip install .


Usage
^^^^^

The best way to learn ScalarStop is to follow the
`Official ScalarStop Tutorial <https://nbviewer.jupyter.org/github/scalarstop/scalarstop/blob/main/notebooks/tutorial.ipynb>`_.

Afterwards, you might want to dig deeper into the ScalarStop documentation.
In general, a typical ScalarStop workflow involves four steps:

1. Organize your datasets with :py:mod:`scalarstop.datablob`.
2. Describe your machine learning model architectures using :py:mod:`scalarstop.model_template`.
3. Load, train, and save machine learning models with :py:mod:`scalarstop.model`.
4. Save hyperparameters and training metrics to a SQLite or PostgreSQL database using :py:mod:`scalarstop.train_store`.
