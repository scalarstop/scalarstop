PYTHON ?= python3
POETRY ?= $(PYTHON) -m poetry
MAKE ?= make
CURL ?= curl --fail --show-error --location

help:
	@cat .makefile-help

clean:
	$(MAKE) -C docs clean
	rm -rf .coverage htmlcov

shell:
	$(POETRY) shell

rm-venv:
	rm -rf $$($(POETRY) env info --path)

install:
	$(POETRY) update
	$(POETRY) install --extras=tensorflow --extras=psycopg2-binary
	$(POETRY) run pip install -r .readthedocs-requirements.txt

install-gpu:
	$(POETRY) update
	$(POETRY) install --extras=tensorflow-gpu --extras=psycopg2-binary
	$(POETRY) run pip install -r .readthedocs-requirements.txt

fmt:
	$(POETRY) run isort --atomic . docs/conf.py
	$(POETRY) run black . docs/conf.py

fmt-check:
	$(POETRY) run isort --check-only . docs/conf.py
	$(POETRY) run black --check . docs/conf.py

lint:
	$(POETRY) run pylint scalarstop tests
	$(POETRY) run mypy scalarstop tests

docs:
	$(POETRY) run $(MAKE) -C docs dirhtml

build:
	$(POETRY) run python -m pip --no-cache-dir wheel --no-deps .

test:
	$(POETRY) run python -m unittest

test-with-coverage:
	$(POETRY) run coverage run --source ./scalarstop --branch -m unittest

coverage-html:
	$(POETRY) run coverage html --ignore-errors

update-intersphinx:
	$(CURL) https://docs.python.org/3.8/objects.inv --output ./docs/_intersphinx-mappings/python.inv
	$(CURL) https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv --output ./docs/_intersphinx-mappings/tensorflow.inv
	$(CURL) https://pandas.pydata.org/pandas-docs/dev/objects.inv --output ./docs/_intersphinx-mappings/pandas.inv
	$(CURL) https://docs.sqlalchemy.org/en/14/objects.inv --output ./docs/_intersphinx-mappings/sqlalchemy.inv

.PHONY: help clean install fmt fmt-check lint docs build test
