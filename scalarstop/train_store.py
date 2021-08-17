"""
Persists :py:class:`~scalarstop.datablob.DataBlob`,
:py:class:`~scalarstop.model_template.ModelTemplate`,
and :py:class:`~scalarstop.model.Model` metadata to a database.

What database should I use?
----------------------------

Currently the :py:class:`TrainStore` supports saving metadata
and metrics to either a SQLite or a PostgreSQL database.
If you are doing all of your work on a single machine, a
SQLite database is easier to set up. But if you are training machine
learning models on multiple machines, you should use a PostgreSQL
database instead of SQLite. The SQLite database is not optimal
for handling multiple concurrent writes.

How can I extend the :py:class:`TrainStore`?
--------------------------------------------

The :py:class:`TrainStore` does not implement absolutely every
type of query that you might want to perform on your
training metrics. However, we directly expose our SQLAlchemy
engine, connection, and tables in the :py:class:`TrainStore`
attributes :py:attr:`TrainStore.engine`,
:py:attr:`TrainStore.connection`, and
:py:attr:`TrainStore.table`.
"""
import dataclasses as _python_dataclasses
import datetime
import sqlite3
import urllib.parse
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import pandas as pd
import sqlalchemy.dialects.postgresql
import sqlalchemy.dialects.sqlite
from log_with_context import Logger
from sqlalchemy import JSON as default_JSON
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    and_,
    asc,
    create_engine,
    desc,
    func,
)
from sqlalchemy import insert as default_insert
from sqlalchemy import select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from scalarstop._datetime import utcnow
from scalarstop.exceptions import SQLite_JSON_ModeDisabled
from scalarstop.hyperparams import enforce_dict

_LOGGER = Logger(__name__)

_TABLE_NAME_PREFIX = "scalarstop__"

_datablob_value_error = ValueError(
    "You should not set both `datablob_name` and `datablob_group_name`. "
    "Set `datablob_name` if you want to be more specific in your search "
    "or set `datablob_group_name` if you want to be more general."
)

_model_template_value_error = ValueError(
    "You should not set both `model_template_name` and `model_template_group_name`. "
    "Set `model_template_name` if you want to be more specific in your search "
    "or set `model_template_group_name` if you want to be more general."
)

_metric_direction_value_error = ValueError(
    "Please provide both `metric_name` and `metric_direction` or neither."
)


@_python_dataclasses.dataclass(frozen=True)
class _ModelMetadata:
    """A dataclass to store :py:class:`TrainStore` metadata for a single :py:class:`~scalarstop.model.Model`."""  # pylint: disable=line-too-long

    model_name: str
    model_class_name: str
    model_epoch_metrics: Dict[str, float]
    model_last_modified: datetime.datetime
    datablob_name: str
    datablob_group_name: str
    datablob_hyperparams: Dict[str, Any]
    model_template_name: str
    model_template_group_name: str
    model_template_hyperparams: Dict[str, Any]
    sort_metric_name: str
    sort_metric_value: float


def _enforce_list(value: Union[str, Sequence[Any]]) -> List[Any]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _censor_sqlalchemy_url_password(url: str) -> str:
    """
    Returns a SQLAlchemy URL without the password.

    Borrowed from https://github.com/WFT/aws-xray-sdk-python/blob/0134819b618962a201f6cb3453fdd24e412046a8/aws_xray_sdk/ext/sqlalchemy/util/decorators.py
    """  # pylint: disable=line-too-long
    u = urllib.parse.urlparse(url)
    # Add Scheme to uses_netloc or // will be missing from url.
    urllib.parse.uses_netloc.append(u.scheme)
    safe_url = ""
    if u.password is None:
        safe_url = u.geturl()
    else:
        # Strip password from URL
        host_info = u.netloc.rpartition("@")[-1]
        parts = u._replace(netloc="{}@{}".format(u.username, host_info))
        safe_url = parts.geturl()
    return safe_url


def _sqlite_json_enabled() -> bool:
    """Return True if this Python installation supports SQLite3 JSON1."""
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT JSON(\'{"a": "b"}\')')
    except sqlite3.OperationalError:
        return False
    cursor.close()
    connection.close()
    return True


def _flatten_hyperparam_results(results) -> List[Dict[str, str]]:
    results_dicts = []
    for row in results:
        row_dict = dict(row)
        dbh = row_dict.pop("datablob_hyperparams")
        mth = row_dict.pop("model_template_hyperparams")
        results_dicts.append(
            dict(
                **row_dict,
                **{f"dbh__{key}": val for key, val in dbh.items()},
                **{f"mth__{key}": val for key, val in mth.items()},
            )
        )
    return results_dicts


class _TrainStoreTables:
    """Manages our :py:class:`sqlalchemy.schema.Table` objects."""

    def __init__(self, table_name_prefix: str, dialect: str):
        """
        Args:
            table_name_prefix: A string prefix to add to all of
                the table names we generate. This allows multiple
                installations of ScalarStop to share the same
                database.

            dialect: The SQLAlchemy dialect that we are
                constructing database tables for.
        """
        if dialect == "postgresql":
            JSON = getattr(sqlalchemy.dialects, dialect).JSONB
        else:
            JSON = default_JSON
        self._table_name_prefix = table_name_prefix
        self._metadata = MetaData()
        self._datablob_table = Table(
            self._table_name_prefix + "datablob",
            self._metadata,
            Column("datablob_name", Text, primary_key=True, nullable=False),
            Column("datablob_group_name", Text, nullable=False),
            Column("datablob_hyperparams", JSON, nullable=False),
            Column(
                "datablob_last_modified",
                DateTime(timezone=True),
                default=utcnow,
                onupdate=utcnow,
                nullable=False,
            ),
            extend_existing=True,
        )

        self._model_template_table = Table(
            self._table_name_prefix + "model_template",
            self._metadata,
            Column("model_template_name", Text, primary_key=True, nullable=False),
            Column("model_template_group_name", Text, nullable=False),
            Column("model_template_hyperparams", JSON, nullable=False),
            Column(
                "model_template_last_modified",
                DateTime(timezone=True),
                default=utcnow,
                onupdate=utcnow,
                nullable=False,
            ),
            extend_existing=True,
        )

        self._model_table = Table(
            self._table_name_prefix + "model",
            self._metadata,
            Column("model_name", Text, primary_key=True, nullable=False),
            Column("model_class_name", Text, nullable=False),
            Column(
                "model_last_modified",
                DateTime(timezone=True),
                default=utcnow,
                onupdate=utcnow,
                nullable=False,
            ),
            Column(
                "datablob_name",
                Text,
                ForeignKey(
                    self._datablob_table.c.datablob_name,
                    onupdate="CASCADE",
                    ondelete="CASCADE",
                ),
                nullable=False,
            ),
            Column(
                "model_template_name",
                Text,
                ForeignKey(
                    self._model_template_table.c.model_template_name,
                    onupdate="CASCADE",
                    ondelete="CASCADE",
                ),
                nullable=False,
            ),
            extend_existing=True,
        )

        self._model_epoch_table = Table(
            self._table_name_prefix + "model_epoch",
            self._metadata,
            Column(
                "model_epoch_id",
                Integer,
                primary_key=True,
                autoincrement=True,
                nullable=False,
            ),
            Column("model_epoch_num", Integer, nullable=False),
            Column(
                "model_name",
                Text,
                ForeignKey(
                    self._model_table.c.model_name,
                    onupdate="CASCADE",
                    ondelete="CASCADE",
                ),
                nullable=False,
            ),
            Column("model_epoch_metrics", JSON, nullable=False),
            Column(
                "model_epoch_last_modified",
                DateTime(timezone=True),
                default=utcnow,
                onupdate=utcnow,
                nullable=False,
            ),
            UniqueConstraint("model_epoch_num", "model_name"),
            extend_existing=True,
        )

    @property
    def metadata(self):
        """The :py:class:`sqlalchemy.sql.schema.MetaData` for this database connection."""
        return self._metadata

    @property
    def datablob(self) -> Table:
        """
        The :py:class:`sqlalchemy.schema.Table` used to store
        :py:class:`~scalarstop.DataBlob` objects.
        """
        return self._datablob_table

    @property
    def model_template(self) -> Table:
        """
        The :py:class:`sqlalchemy.schema.Table`used to store
        :py:class:`~scalarstop.ModelTemplate` objects.
        """
        return self._model_template_table

    @property
    def model(self) -> Table:
        """
        The :py:class:`sqlalchemy.schema.Table` used to store
        :py:class:`~scalarstop.model.Model` objects.
        """
        return self._model_table

    @property
    def model_epoch(self) -> Table:
        """
        The :py:class:`sqlalchemy.schema.Table` used to store
        metrics from model epochs.
        """
        return self._model_epoch_table


class TrainStore:
    """Loads and saves names, hyperparameters, and training metrics from :py:class:`~scalarstop.datablob.DataBlob`, :py:class:`~scalarstop.model_template.ModelTemplate`, and :py:class:`~scalarstop.model.Model` objects."""  # pylint: disable=line-too-long

    @classmethod
    def from_filesystem(
        cls,
        *,
        filename: str,
        table_name_prefix: Optional[str] = None,
        echo: bool = False,
    ) -> "TrainStore":
        """
        Use a SQLite3 database file on the local filesystem as the train store.

        Args:
            filename: The filename of the SQLite3 file.

            table_name_prefix: A string prefix to add to all of
                the table names we generate. This allows multiple
                installations of ScalarStop to share the same
                database.

            echo: Set to ``True`` to print out the SQL statements
                that the :py:class:`TrainStore` executes.
        """
        return cls(
            connection_string="sqlite:///" + filename,
            table_name_prefix=table_name_prefix,
            echo=echo,
        )

    def __init__(
        self,
        connection_string: str,
        *,
        table_name_prefix: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Create a :py:class:`TrainStore` instance connected to
        an external database.

        Use this constructor if you want to connect to
        a PostgreSQL database. If you want to use a SQLite file as the database,
        you should instead use the :py:meth:`TrainStore.from_filesystem`
        classmethod.

        Args:
            connection_string: A SQLAlchemy database connection
                string for connecting to a database. A typical
                PostgreSQL connection string looks like
                ``"postgresql://username:password@hostname:port/database"``,
                with the ``port`` defaulting to ``5432``.

            table_name_prefix: A string prefix to add to all of
                the table names we generate. This allows multiple
                installations of ScalarStop to share the same
                database.

            echo: Set to ``True`` to print out the SQL statements
                that the :py:class:`TrainStore` executes.
        """
        # The `postgres://` connection prefix is deprecated after
        # SQLAlchemy 1.4. We'll do the find-and-replace on the user's
        # behalf.
        if connection_string.startswith("postgres://"):
            connection_string = connection_string.replace(
                "postgres://", "postgresql://", 1
            )
        self._connection_string = connection_string
        self._connection_string_no_password = _censor_sqlalchemy_url_password(
            self._connection_string
        )
        self._table_name_prefix = table_name_prefix or _TABLE_NAME_PREFIX
        self._echo = echo
        self._engine = create_engine(
            self._connection_string,
            echo=self._echo,
            future=True,
        )
        self._connection = self._engine.connect()
        if self._engine.name == "sqlite":
            if not _sqlite_json_enabled():
                raise SQLite_JSON_ModeDisabled()
            with self._connection.begin():
                self._connection.execute(text("PRAGMA foreign_keys = ON"))
        self._table = _TrainStoreTables(
            table_name_prefix=self._table_name_prefix, dialect=self._engine.name
        )
        with self._connection.begin():
            self._table.metadata.create_all(bind=self._engine)

    def __repr__(self) -> str:
        return f"<sp.TrainStore {self._connection_string_no_password}>"

    @property
    def table(self) -> _TrainStoreTables:
        """
        References to the :py:class:`sqlalchemy.schema.Table` objects
        representing our database tables.

        Currently, there are four tables that are attributes to this
        property:

        * ``datablob``
        * ``model_template``
        * ``model``
        * ``model_epoch``
        """
        return self._table

    @property
    def engine(self) -> Engine:
        """
        The currently active :py:class:`sqlalchemy.engine.Engine`.

        This is useful if you want to write custom SQLAlchemy
        code on top of :py:class:`TrainStore`.
        """
        return self._engine

    @property
    def connection(self) -> Connection:
        """
        The currently active :py:class:`sqlalchemy.engine.Connection`.

        This is useful if you want to write custom SQLAlchemy
        code on top of :py:class:`TrainStore`.
        """
        return self._connection

    def _insert(
        self, *, table, values, index_elements=None, ignore_existing: bool
    ) -> None:
        if ignore_existing:
            if self._engine.name == "sqlite":
                with self.connection.begin():
                    self.connection.execute(
                        sqlalchemy.dialects.sqlite.insert(table)
                        .values(**values)
                        .on_conflict_do_nothing(index_elements=index_elements)
                    )
            elif self._engine.name == "postgresql":
                with self.connection.begin():
                    self.connection.execute(
                        sqlalchemy.dialects.postgresql.insert(table)
                        .values(**values)
                        .on_conflict_do_nothing(index_elements=index_elements)
                    )
            else:
                try:
                    with self.connection.begin():
                        self.connection.execute(default_insert(table).values(**values))
                except IntegrityError:
                    _LOGGER.info("Suppressed IntegrityError", exc_info=True)
        else:
            with self.connection.begin():
                self.connection.execute(default_insert(table).values(**values))

    def _as_pandas(self, stmt) -> pd.DataFrame:
        with self.connection.begin():
            return pd.read_sql_query(sql=stmt, con=self.connection)

    def insert_datablob(self, datablob, *, ignore_existing: bool = False) -> None:
        """
        Logs the :py:class:`~scalarstop.datablob.DataBlob` name, group name,
        and hyperparams to the :py:class:`TrainStore`.

        Args:
            datablob: A :py:class:`~scalarstop.datablob.DataBlob`
                instance whose name and hyperparameters that
                we want to record in the database.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.datablob.DataBlob`
                with the same name is already in the database,
                in which case this function will do nothing.
                Note that :py:class:`~scalarstop.datablob.DataBlob`
                instances are supposed to be immutable, so
                :py:class:`TrainStore` does not implement
                updating them.
        """
        self.insert_datablob_by_str(
            name=datablob.name,
            group_name=datablob.group_name,
            hyperparams=datablob.hyperparams,
            ignore_existing=ignore_existing,
        )

    def insert_datablob_by_str(
        self,
        *,
        name: str,
        group_name: str,
        hyperparams: Any,
        ignore_existing: bool = False,
    ):
        """
        Logs the :py:class:`~scalarstop.datablob.DataBlob` name, group
        name, and hyperparams to the :py:class:`TrainStore`.

        Args:
            name: Your :py:class:`~scalarstop.datablob.DataBlob`
                name.

            group_name: Your :py:class:`~scalarstop.datablob.DataBlob`
                group name.

            hyperparams: Your :py:class:`~scalarstop.datablob.DataBlob`
                hyperparameters.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.datablob.DataBlob`
                with the same name is already in the database,
                in which case this function will do nothing.
                Note that :py:class:`~scalarstop.datablob.DataBlob`
                instances are supposed to be immutable, so
                :py:class:`TrainStore` does not implement
                updating them.
        """
        values = dict(
            datablob_name=name,
            datablob_group_name=group_name,
            datablob_hyperparams=enforce_dict(hyperparams),
        )
        self._insert(
            table=self.table.datablob,
            values=values,
            index_elements=[self.table.datablob.c.datablob_name],
            ignore_existing=ignore_existing,
        )

    def _query_datablob_stmt(
        self,
        *,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        stmt = select(
            [
                self.table.datablob.c.datablob_name.label("name"),
                self.table.datablob.c.datablob_group_name.label("group_name"),
                self.table.datablob.c.datablob_hyperparams.label("hyperparams"),
                self.table.datablob.c.datablob_last_modified.label("last_modified"),
            ]
        ).select_from(self.table.datablob)

        if datablob_name and datablob_group_name:
            raise _datablob_value_error
        if datablob_name:
            stmt = stmt.where(
                self.table.datablob.c.datablob_name.in_(_enforce_list(datablob_name))
            )
        elif datablob_group_name:
            stmt = stmt.where(
                self.table.datablob.c.datablob_group_name.in_(
                    _enforce_list(datablob_group_name)
                )
            )

        stmt = stmt.order_by(self.table.datablob.c.datablob_last_modified)
        return stmt

    def list_datablobs(
        self,
        *,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """
        Returns a :py:class:`pandas.DataFrame` listing the
        :py:class:`~scalarstop.datablob.DataBlob` names in the database.

        If you call this method without any arguments, it will list
        ALL of the :py:class:`~scalarstop.datablob.DataBlob` s in
        the database. You can narrow down your results by providing
        ONE (but not both) of the below arguments.

        Args:
            datablob_name: Either a single :py:class:`~scalarstop.datablob.DataBlob`
                name or a list of names to select.

            datablob_group_name: Either a single :py:class:`~scalarstop.datablob.DataBlob`
                group name or a list of group names to select.
        """
        return self._as_pandas(
            self._query_datablob_stmt(
                datablob_name=datablob_name, datablob_group_name=datablob_group_name
            )
        )

    def insert_model_template(self, model_template, *, ignore_existing: bool = False):
        """
        Logs the :py:class:`~scalarstop.model_template.ModelTemplate`
        name, group name, and hyperparams to the :py:class:`TrainStore`.

        Args:
            model_template: A :py:class:`~scalarstop.model_template.ModelTemplate`
                instance whose name and hyperparameters that
                we want to record in the database.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.model_template.ModelTemplate`
                with the same name is already in the database,
                in which case this function will do nothing.
                Note that :py:class:`~scalarstop.model_template.ModelTemplate`
                instances are supposed to be immutable, so
                :py:class:`TrainStore` does not implement
                updating them.
        """
        self.insert_model_template_by_str(
            name=model_template.name,
            group_name=model_template.group_name,
            hyperparams=model_template.hyperparams,
            ignore_existing=ignore_existing,
        )

    def insert_model_template_by_str(
        self, *, name: str, group_name: str, hyperparams, ignore_existing: bool = False
    ):
        """
        Logs the :py:class:`~scalarstop.model_template.ModelTemplate`
        name, group name, and hyperparams to the :py:class:`TrainStore`.

        Args:
            name: Your :py:class:`~scalarstop.model_template.ModelTemplate`
                name.

            group_name: Your :py:class:`~scalarstop.model_template.ModelTemplate`
                group name.

            hyperparams: Your :py:class:`~scalarstop.model_template.ModelTemplate`
                hyperparameters.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.model_template.ModelTemplate`
                with the same name is already in the database,
                in which case this function will do nothing.
                Note that :py:class:`~scalarstop.model_template.ModelTemplate`
                instances are supposed to be immutable, so
                :py:class:`TrainStore` does not implement
                updating them.
        """
        values = dict(
            model_template_name=name,
            model_template_group_name=group_name,
            model_template_hyperparams=enforce_dict(hyperparams),
        )
        self._insert(
            table=self.table.model_template,
            values=values,
            index_elements=[self.table.model_template.c.model_template_name],
            ignore_existing=ignore_existing,
        )

    def _query_model_template_stmt(
        self,
        *,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        stmt = select(
            [
                self.table.model_template.c.model_template_name.label("name"),
                self.table.model_template.c.model_template_group_name.label(
                    "group_name"
                ),
                self.table.model_template.c.model_template_hyperparams.label(
                    "hyperparams"
                ),
                self.table.model_template.c.model_template_last_modified.label(
                    "last_modified"
                ),
            ]
        ).select_from(self.table.model_template)

        if model_template_name and model_template_group_name:
            raise _model_template_value_error
        if model_template_name:
            stmt = stmt.where(
                self.table.model_template.c.model_template_name.in_(
                    _enforce_list(model_template_name)
                )
            )
        elif model_template_group_name:
            stmt = stmt.where(
                self.table.model_template.c.model_template_group_name.in_(
                    _enforce_list(model_template_group_name)
                )
            )
        stmt = stmt.order_by(self.table.model_template.c.model_template_last_modified)
        return stmt

    def list_model_templates(
        self,
        *,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        """
        Returns a :py:class:`pandas.DataFrame` listing ALL of the rows in the
        :py:class:`~scalarstop.model_template.ModelTemplate` table.

        If you call this method without any arguments, it will list
        ALL of the :py:class:`~scalarstop.model_template.ModelTemplate` s in
        the database. You can narrow down your results by providing
        ONE (but not both) of the below arguments.

        Args:
            model_template_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                name or a list of names to select.

            model_template_group_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                group name or a list of group names to select.
        """
        return self._as_pandas(
            self._query_model_template_stmt(
                model_template_name=model_template_name,
                model_template_group_name=model_template_group_name,
            )
        )

    def insert_model(self, model, *, ignore_existing: bool = False):
        """
        Logs the :py:class:`~scalarstop.model.Model` name,
        :py:class:`~scalarstop.datablob.DataBlob`, and
        :py:class;`~scalarstop.model_template.ModelTemplate`
        to the :py:class:`TrainStore`.

        Args:
            model: A :py:class:`~scalarstop.model.Model`
                instance whose name and hyperparameters that
                we want to record in the database.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.model.Model`
                with the same name is already in the database,
                in which case this function will do nothing.
                The :py:class:`TrainStore` does not implement
                the updating of :py:class:`~scalarstop.model.Model`
                name or hyperparameters. The only way to change
                a :py:class:`~scalarstop.model.Model` is to
                log more epochs.
        """
        self.insert_model_by_str(
            name=model.name,
            model_class_name=model.__class__.__name__,
            datablob_name=model.datablob.name,
            model_template_name=model.model_template.name,
            ignore_existing=ignore_existing,
        )

    def insert_model_by_str(
        self,
        *,
        name: str,
        model_class_name: str,
        datablob_name: str,
        model_template_name: str,
        ignore_existing: bool = False,
    ) -> None:
        """
        Logs the :py:class:`~scalarstop.model.Model` name,
        :py:class:`~scalarstop.datablob.DataBlob`, and
        :py:class;`~scalarstop.model_template.ModelTemplate`
        to the :py:class:`TrainStore`.

        Args:
            name: The  :py:class:`~scalarstop.model.Model` name.

            model_class_name: The :py:class:`~scalarstop.model.Model`
                subclass name used. If you are using
                :py:class:`~scalarstop.model.KerasModel`,
                then this value is the string ``"KerasModel"``.

            datablob_name: The :py:class:`~scalarstop.datablob.DataBlob`
                name used to create the :py:class:`~scalarstop.model.Model`
                instance.

            model_template_name: The
                :py:class:`~scalarstop.model_template.ModelTemplate`
                name used to create the :py:class:`~scalarstop.model.Model`
                instance.

            ignore_existing: Set this to ``True`` to ignore
                if a :py:class:`~scalarstop.model.Model`
                with the same name is already in the database,
                in which case this function will do nothing.
                The :py:class:`TrainStore` does not implement
                the updating of :py:class:`~scalarstop.model.Model`
                name or hyperparameters. The only way to change
                a :py:class:`~scalarstop.model.Model` is to
                log more epochs.
        """
        values = dict(
            model_name=name,
            model_class_name=model_class_name,
            datablob_name=datablob_name,
            model_template_name=model_template_name,
        )
        self._insert(
            table=self.table.model,
            values=values,
            index_elements=[self.table.model.c.model_name],
            ignore_existing=ignore_existing,
        )

    def _query_model_stmt(
        self,
        *,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        where_conditions = []

        if datablob_name and datablob_group_name:
            raise _datablob_value_error
        if datablob_name:
            where_conditions.append(
                self.table.model.c.datablob_name.in_(_enforce_list(datablob_name))
            )
        elif datablob_group_name:
            where_conditions.append(
                self.table.model.c.datablob_group_name.in_(
                    _enforce_list(datablob_group_name)
                )
            )

        if model_template_name and model_template_group_name:
            raise _model_template_value_error
        if model_template_name:
            where_conditions.append(
                self.table.model.c.model_template_name.in_(
                    _enforce_list(model_template_name)
                )
            )
        elif model_template_group_name:
            where_conditions.append(
                self.table.model.c.model_template_group_name.in_(
                    _enforce_list(model_template_group_name)
                )
            )

        stmt = (
            select(
                [
                    self.table.model.c.model_name,
                    self.table.model.c.model_class_name,
                    self.table.model.c.model_last_modified,
                    self.table.datablob.c.datablob_name,
                    self.table.datablob.c.datablob_group_name,
                    self.table.datablob.c.datablob_hyperparams,
                    self.table.model_template.c.model_template_name,
                    self.table.model_template.c.model_template_group_name,
                    self.table.model_template.c.model_template_hyperparams,
                ]
            )
            .select_from(self.table.model)
            .join(
                self.table.datablob,
                self.table.model.c.datablob_name == self.table.datablob.c.datablob_name,
            )
            .join(
                self.table.model_template,
                self.table.model.c.model_template_name
                == self.table.model_template.c.model_template_name,
            )
        )
        if where_conditions:
            stmt = stmt.where(and_(*where_conditions))
        stmt = stmt.order_by(self.table.model.c.model_last_modified)
        return stmt

    def _query_model_by_epoch_stmt(
        self,
        *,
        metric_name: Optional[str] = None,
        metric_direction: Optional[str] = None,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
        limit: Optional[int] = None,
        return_other_metrics: bool = True,
    ):
        where_conditions = []

        if datablob_name and datablob_group_name:
            raise _datablob_value_error
        if datablob_name:
            where_conditions.append(
                self.table.datablob.c.datablob_name.in_(_enforce_list(datablob_name))
            )
        elif datablob_group_name:
            where_conditions.append(
                self.table.datablob.c.datablob_group_name.in_(
                    _enforce_list(datablob_group_name)
                )
            )

        if model_template_name and model_template_group_name:
            raise _model_template_value_error
        if model_template_name:
            where_conditions.append(
                self.table.model_template.c.model_template_name.in_(
                    _enforce_list(model_template_name)
                )
            )
        elif model_template_group_name:
            where_conditions.append(
                self.table.model_template.c.model_template_group_name.in_(
                    _enforce_list(model_template_group_name)
                )
            )

        columns = [
            self.table.model.c.model_name,
            self.table.model.c.model_class_name,
            self.table.model.c.model_last_modified,
            self.table.datablob.c.datablob_name,
            self.table.datablob.c.datablob_group_name,
            self.table.datablob.c.datablob_hyperparams,
            self.table.model_template.c.model_template_name,
            self.table.model_template.c.model_template_group_name,
            self.table.model_template.c.model_template_hyperparams,
        ]

        if return_other_metrics:
            columns.append(self.table.model_epoch.c.model_epoch_metrics)

        group_by_columns = columns.copy()
        sort_metric_value = None
        if metric_name and metric_direction:
            if metric_direction == "max":
                select_sorting_func = func.max
                order_by_sorting_func = desc
            elif metric_direction == "min":
                select_sorting_func = func.min
                order_by_sorting_func = asc
            else:
                raise ValueError(
                    "The argument metric_direction should be either 'max' or 'min', "
                    f"not '{metric_direction}."
                )
            sort_metric_value = select_sorting_func(
                self.table.model_epoch.c.model_epoch_metrics[metric_name].as_float()
            ).label("sort_metric_value")

            columns.append(sort_metric_value)
        elif metric_name:
            raise _metric_direction_value_error
        elif metric_direction:
            raise _metric_direction_value_error

        stmt = (
            select(columns)
            .select_from(self.table.model_epoch)
            .join(
                self.table.model,
                self.table.model_epoch.c.model_name == self.table.model.c.model_name,
            )
            .join(
                self.table.datablob,
                self.table.model.c.datablob_name == self.table.datablob.c.datablob_name,
            )
            .join(
                self.table.model_template,
                self.table.model.c.model_template_name
                == self.table.model_template.c.model_template_name,
            )
        )
        if where_conditions:
            stmt = stmt.where(and_(*where_conditions))

        if sort_metric_value is not None:
            stmt = stmt.group_by(*group_by_columns).order_by(
                order_by_sorting_func(sort_metric_value)
            )
        else:
            stmt = stmt.order_by(self.table.model.c.model_last_modified)

        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def list_models(
        self,
        *,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """
        Returns a :py:class:`pandas.DataFrame` listing ALL of the rows in the
        :py:class:`~scalarstop.model.Model` table.

        If you call this method without any arguments, it will list ALL
        of the :py:class:`~scalarstop.model.Model` s in the database.
        Optionally, you can narrow down the results with the following
        values.

        Note that you can provide either ``datablob_name`` or
        ``datablob_group_name``, but not both.

        Similarly, you can provide either ``model_template_name``
        or ``model_template_group_name``, but not both.

        Args:
            datablob_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                name or a list of names to select.

            datablob_group_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                group name or a list of group names to select.

            model_template_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                name or a list of names to select.

            model_template_group_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                group name or a list of group names to select.
        """
        stmt = self._query_model_stmt(
            datablob_name=datablob_name,
            datablob_group_name=datablob_group_name,
            model_template_name=model_template_name,
            model_template_group_name=model_template_group_name,
        )
        with self.connection.begin():
            results = self.connection.execute(stmt)
            results_dicts = _flatten_hyperparam_results(results)
        return pd.DataFrame(results_dicts)

    def list_models_grouped_by_epoch_metric(
        self,
        *,
        metric_name: str,
        metric_direction: str,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """
        Returns a :py:class:`pandas.DataFrame` listing ALL of the rows in the
        :py:class:`~scalarstop.model.Model` table AND a metric from
        the model's best-performing epoch.

        You provide this method with a model epoch metric name
        and whether to maximize or minimize this, and then
        it returns all of the models and the best metric value.

        Note that you can provide either ``datablob_name`` or
        ``datablob_group_name``, but not both.

        Similarly, you can provide either ``model_template_name``
        or ``model_template_group_name``, but not both.

        Args:
            metric_name: The name of one of the metrics
                tracked when training a model. This might be a value
                like ``"loss"`` or ``"val_accuracy"``.

            metric_direction: Set this to ``"min"`` if the metric
                you picked in ``metric_name`` is a value where
                lower values are better--such as ``"loss"``.
                Set this to ``"max"`` if higher values of your
                metric are better--such as ``"accuracy"``.

            datablob_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                name or a list of names to select.

            datablob_group_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                group name or a list of group names to select.

            model_template_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                name or a list of names to select.

            model_template_group_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                group name or a list of group names to select.

        Returns a :py:class:`pandas.DataFrame` with the following
        columns:

        * ``model_name``
        * ``model_class_name``
        * ``model_last_modified``
        * ``datablob_name``
        * ``datablob_group_name``
        * ``model_template_name``
        * ``model_template_group_name``
        * ``sort_metric_value``
        * :py:class:`~scalarstop.model_template.ModelTemplate`
          hyperparameter names prefixed with ``mth__``
        * :py:class:`~scalarstop.datablob.DataBlob`
          hyperparameter names prefixed with ``dbh__``

        """
        stmt = self._query_model_by_epoch_stmt(
            metric_name=metric_name,
            metric_direction=metric_direction,
            datablob_name=datablob_name,
            datablob_group_name=datablob_group_name,
            model_template_name=model_template_name,
            model_template_group_name=model_template_group_name,
            return_other_metrics=False,
        )
        with self.connection.begin():
            results_dicts = _flatten_hyperparam_results(
                self.connection.execute(stmt).fetchall()
            )
        return pd.DataFrame(results_dicts)

    def insert_model_epoch(
        self, *, epoch_num: int, model_name: str, metrics, ignore_existing: bool = False
    ) -> None:
        """
        Logs a new epoch for a :py:class:`~scalarstop.model.Model`
        to the :py:class:`TrainStore`.

        Args:
            epoch_num: The epoch number that we are adding.

            model_name: The name of the :py:class:`~scalarstop.model.Model`
                tha we are training.

            metrics: A dictionary of metric names and values
                to save.

            ignore_existing: Set this to ``True`` to ignore
                if the database already has a row with the same
                ``(model_name, epoch_num)`` pair.
        """
        values = dict(
            model_epoch_num=epoch_num,
            model_name=model_name,
            model_epoch_metrics=enforce_dict(metrics),
        )
        self._insert(
            table=self.table.model_epoch,
            values=values,
            index_elements=[
                self.table.model_epoch.c.model_epoch_num,
                self.table.model_epoch.c.model_name,
            ],
            ignore_existing=ignore_existing,
        )

    def bulk_insert_model_epochs(self, model) -> None:
        """
        Insert a list of :py:class:`~scalarstop.model.Model` epochs at once.

        This method will politely ignore if the database already
        contains rows with the same model name and epoch number.

        Currently this method only works if you are using
        either SQLite or PostgreSQL as the backing database.

        Args:
            model: The :py:class:`~scalarstop.model.Model`
                with the epochs that we want to save.
        """
        values = []
        for model_epoch_num in range(model.current_epoch):
            current_epoch = dict(
                model_epoch_num=model_epoch_num,
                model_name=model.name,
                model_epoch_metrics=dict(),
            )
            for metric_name, metric_values in model.history.items():
                try:
                    current_metric_value = metric_values[model_epoch_num]
                except IndexError:
                    continue
                else:
                    current_epoch["model_epoch_metrics"][
                        metric_name
                    ] = current_metric_value
            values.append(current_epoch)
        if self._engine.name == "sqlite" or self._engine.name == "postgresql":
            insert = getattr(sqlalchemy.dialects, self._engine.name).insert
            stmt = insert(self.table.model_epoch).on_conflict_do_nothing(
                index_elements=[
                    self.table.model_epoch.c.model_epoch_num,
                    self.table.model_epoch.c.model_name,
                ]
            )
            with self.connection.begin():
                self.connection.execute(stmt, values)
        else:
            raise NotImplementedError(
                "Bulk insert of model epochs is currently only "
                "supported for SQLite and PostgreSQL."
            )

    def _query_model_epoch_stmt(
        self,
        *,
        model_name: Optional[Union[str, Sequence[str]]] = None,
    ):
        stmt = select(
            [
                self.table.model_epoch.c.model_epoch_num.label("epoch_num"),
                self.table.model_epoch.c.model_name,
                self.table.model_epoch.c.model_epoch_last_modified.label(
                    "last_modified"
                ),
                self.table.model_epoch.c.model_epoch_metrics.label("metrics"),
            ]
        ).select_from(self.table.model_epoch)
        if model_name:
            stmt = stmt.where(
                self.table.model_epoch.c.model_name.in_(_enforce_list(model_name))
            )
        stmt = stmt.order_by(self.table.model_epoch.c.model_epoch_num)
        return stmt

    def list_model_epochs(
        self,
        model_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """
        Returns a :py:class:`pandas.DataFrame` listing
        :py:class:`~scalarstop.model.Model` epochs.

        By default, this lists ALL epochs in the database for ALL
        models. You can narrow down the search with the following
        arguments.

        Args:
            model_name: Specify a single model name or a list
                of model names whose epochs we are interested in.
        """
        results_dicts = []
        with self.connection.begin():
            results = self.connection.execute(
                self._query_model_epoch_stmt(model_name=model_name)
            )
            for row in results:
                row_dict = dict(row)
                metrics = row_dict.pop("metrics")
                results_dicts.append(
                    dict(
                        **row_dict,
                        **{f"metric__{key}": val for key, val in metrics.items()},
                    )
                )
        return pd.DataFrame(results_dicts)

    def get_current_epoch(self, model_name: str) -> int:
        """
        Returns how many epochs a given :py:class:`~scalarstop.model.Model` has been
        trained for.

        Returns 0 if the given model is not registered in the
        :py:class:`TrainStore`.

        This information is also saved in the directory created when a
        :py:class:`~scalarstop.model.Model` instance is saved to the filesystem
        and is available in the attribute
        :py:attr:`~scalarstop.model.Model.current_epoch`.
        """
        if not isinstance(model_name, str):
            raise ValueError(
                "TrainStore.get_current_epoch() only takes a single "
                "model name as a string. You cannot supply a list "
                f"of model names. You provided {model_name=}"
            )
        stmt = (
            select(
                func.max(self.table.model_epoch.c.model_epoch_num).label("epoch_num")
            )
            .select_from(self.table.model_epoch)
            .where(self.table.model_epoch.c.model_name == model_name)
        )
        with self.connection.begin():
            current_epoch = self.connection.execute(stmt).fetchone()[0]
        if current_epoch is None:
            return 0
        return current_epoch

    def get_best_model(
        self,
        *,
        metric_name: str,
        metric_direction: str,
        datablob_name: Optional[Union[str, Sequence[str]]] = None,
        datablob_group_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_name: Optional[Union[str, Sequence[str]]] = None,
        model_template_group_name: Optional[Union[str, Sequence[str]]] = None,
    ) -> _ModelMetadata:
        """
        Return metadata about the model with the best performance on a metric.

        This method queries the database, looking for the
        :py:class:`~scalarstop.model.Model` with the best performance
        on the metric you specified in the parameter ``metric_name``.
        By default, this returns ALL models in the database
        sorted by your metric name. Most likely, you will want
        to narrow down your search using the below arguments.

        Note that you can provide either ``datablob_name`` or
        ``datablob_group_name``, but not both.

        Similarly, you can provide either ``model_template_name``
        or ``model_template_group_name``, but not both.

        Args:
            metric_name: The name of one of the metrics
                tracked when training a model. This might be a value
                like ``"loss"`` or ``"val_accuracy"``.

            metric_direction: Set this to ``"min"`` if the metric
                you picked in ``metric_name`` is a value where
                lower values are better--such as ``"loss"``.
                Set this to ``"max"`` if higher values of your
                metric are better--such as ``"accuracy"``.

            datablob_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                name or a list of names to select.

            datablob_group_name: Either a single
                :py:class:`~scalarstop.datablob.DataBlob`
                group name or a list of group names to select.

            model_template_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                name or a list of names to select.

            model_template_group_name: Either a single
                :py:class:`~scalarstop.model_template.ModelTemplate`
                group name or a list of group names to select.

        Returns a dataclass with the following attributes:
            * ``model_name``
            * ``model_class_name``
            * ``model_epoch_metrics``
            * ``model_last_modified``
            * ``datablob_name``
            * ``datablob_group_name``
            * ``datablob_hyperparams``
            * ``model_template_name``
            * ``model_template_group_name``
            * ``model_template_hyperparams``
            * ``sort_metric_name``
            * ``sort_metric_value``
        """
        stmt = self._query_model_by_epoch_stmt(
            metric_name=metric_name,
            metric_direction=metric_direction,
            datablob_name=datablob_name,
            datablob_group_name=datablob_group_name,
            model_template_name=model_template_name,
            model_template_group_name=model_template_group_name,
            limit=1,
        )
        with self.connection.begin():
            result_dict = dict(self.connection.execute(stmt).fetchone())

        the_dataclass = _ModelMetadata(
            **result_dict,
            sort_metric_name=metric_name,
        )
        return the_dataclass

    def __enter__(self) -> "TrainStore":
        return self

    def close(self) -> None:
        """
        Close the database connection.

        This is also called by the context manager's ``__exit__()`` method.
        """
        self.connection.close()

    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        self.close()
        return False
