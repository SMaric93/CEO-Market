"""WRDS client adapters used to pull subscription-backed source tables.

The empirical package never requires WRDS at runtime, but many researchers do want a
bootstrap layer that can fetch Compustat, CRSP, Execucomp, and BoardEx tables directly
when their institution licenses them. This module keeps that dependency optional and
isolated from the rest of the pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from whogetsconsidered.config import WhoGetsConsideredConfig


class WrdsClientProtocol(Protocol):
    """Minimal protocol needed by the WRDS pull pipeline."""

    def get_table(
        self,
        *,
        library: str,
        table: str,
        columns: list[str] | None = None,
        rows: int | None = None,
    ) -> Any:
        """Fetch a WRDS table into a dataframe-like object."""

    def raw_sql(self, sql: str, *, date_cols: list[str] | None = None) -> Any:
        """Run a SQL query against WRDS and return a dataframe-like object."""

    def close(self) -> None:
        """Close the underlying WRDS connection."""


class WrdsConnectionAdapter:
    """Thin runtime adapter around `wrds.Connection` with version-tolerant fallbacks."""

    def __init__(self, username: str | None = None, *, pgpass_file: str | None = None) -> None:
        try:
            import wrds  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise ImportError(
                "WRDS support is optional. Install it with `pip install -e .[wrds]`."
            ) from exc

        self._previous_pgpassfile = os.environ.get("PGPASSFILE")
        if pgpass_file is not None:
            os.environ["PGPASSFILE"] = pgpass_file

        if username is None:
            self._connection = wrds.Connection()
            return

        try:
            self._connection = wrds.Connection(wrds_username=username)
        except TypeError:
            self._connection = wrds.Connection(username=username)

    def get_table(
        self,
        *,
        library: str,
        table: str,
        columns: list[str] | None = None,
        rows: int | None = None,
    ) -> Any:
        """Fetch a WRDS table, adapting to API variants across package versions."""

        kwargs: dict[str, Any] = {"library": library, "table": table}
        if columns:
            kwargs["columns"] = columns
        if rows is not None:
            try:
                return self._connection.get_table(rows=rows, **kwargs)
            except TypeError:
                return self._connection.get_table(obs=rows, **kwargs)
        return self._connection.get_table(**kwargs)

    def raw_sql(self, sql: str, *, date_cols: list[str] | None = None) -> Any:
        """Execute raw SQL against WRDS."""

        kwargs: dict[str, Any] = {}
        if date_cols:
            kwargs["date_cols"] = date_cols
        return self._connection.raw_sql(sql, **kwargs)

    def close(self) -> None:
        """Close the WRDS connection when the stage finishes."""

        self._connection.close()
        if self._previous_pgpassfile is None:
            os.environ.pop("PGPASSFILE", None)
        else:
            os.environ["PGPASSFILE"] = self._previous_pgpassfile


def build_wrds_client(config: WhoGetsConsideredConfig) -> WrdsClientProtocol:
    """Instantiate the optional WRDS client from typed config."""

    pgpass_file = None if config.wrds.pgpass_file is None else str(config.wrds.pgpass_file)
    return WrdsConnectionAdapter(username=config.wrds.username, pgpass_file=pgpass_file)
