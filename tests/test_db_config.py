from __future__ import annotations

import pytest

from tw_stock_ai.db import normalize_database_url


def test_normalize_database_url_rejects_placeholder_tokens() -> None:
    with pytest.raises(RuntimeError, match="placeholder"):
        normalize_database_url("postgresql://user:pass@host:<port>/db")


def test_normalize_database_url_rewrites_postgres_scheme() -> None:
    assert (
        normalize_database_url("postgres://user:pass@host:5432/db")
        == "postgresql+psycopg://user:pass@host:5432/db"
    )


def test_normalize_database_url_allows_sqlite() -> None:
    assert normalize_database_url("sqlite:///./data/app.db") == "sqlite:///./data/app.db"
