PYTHON ?= python3
POETRY ?= $(PYTHON) -m poetry
MAKE ?= make
CURL ?= curl --fail --show-error --location

help:
	@cat .makefile-help
.PHONY: help

clean:
	$(MAKE) -C docs clean
	rm -rf .coverage htmlcov
.PHONY: clean

shell:
	$(POETRY) shell
.PHONY: shell

rm-venv:
	rm -rf $$($(POETRY) env info --path)
.PHONY: rm-venv

update:
	$(POETRY) update
.PHONY: update

install:
	$(POETRY) install --extras=tensorflow --extras=psycopg2-binary
.PHONY: install

install-gpu:
	$(POETRY) install --extras=tensorflow-gpu --extras=psycopg2-binary
.PHONY: install-gpu

install-docs:
	$(POETRY) run pip install -r .readthedocs-requirements.txt
.PHONY: install-docs

fmt:
	$(POETRY) run isort --atomic . docs/conf.py
	$(POETRY) run black . docs/conf.py
.PHONY: fmt

fmt-check:
	$(POETRY) run isort --check-only . docs/conf.py
	$(POETRY) run black --check . docs/conf.py
.PHONY: fmt-check

lint:
	$(POETRY) run pylint scalarstop tests
	$(POETRY) run mypy scalarstop tests
.PHONY: lint

docs:
	$(POETRY) run $(MAKE) -C docs dirhtml
.PHONY: docs

build:
	$(POETRY) run python -m pip --no-cache-dir wheel --no-deps .
.PHONY: build

test:
	$(POETRY) run python -m unittest
.PHONY: test

test-with-coverage:
	$(POETRY) run coverage run --source ./scalarstop --branch -m unittest
.PHONY: test-with-coverage

coverage-html:
	$(POETRY) run coverage html --ignore-errors
.PHONY: coverage-html

update-intersphinx:
	$(CURL) https://docs.python.org/3.8/objects.inv --output ./docs/_intersphinx-mappings/python.inv
	$(CURL) https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv --output ./docs/_intersphinx-mappings/tensorflow.inv
	$(CURL) https://pandas.pydata.org/pandas-docs/dev/objects.inv --output ./docs/_intersphinx-mappings/pandas.inv
	$(CURL) https://docs.sqlalchemy.org/en/14/objects.inv --output ./docs/_intersphinx-mappings/sqlalchemy.inv
.PHONY: update-intersphinx
