from collections.abc import Generator
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import Base

settings = get_settings()


def normalize_database_url(database_url: str) -> str:
    normalized = (database_url or "").strip()
    if not normalized:
        raise RuntimeError("DATABASE_URL is empty")
    if "<" in normalized or ">" in normalized:
        raise RuntimeError(
            "DATABASE_URL still contains placeholder tokens like <host> or <port>. "
            "Use a real connection string, or on Zeabur set DATABASE_URL=${POSTGRES_CONNECTION_STRING}."
        )
    if normalized.startswith("${") and normalized.endswith("}"):
        raise RuntimeError(
            "DATABASE_URL was not expanded by the platform. "
            "On Zeabur, set DATABASE_URL to ${POSTGRES_CONNECTION_STRING} in the Variables UI, "
            "not inside a local .env file committed to the repo."
        )
    if normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql+psycopg://", 1)
    elif normalized.startswith("postgresql://") and not normalized.startswith("postgresql+"):
        normalized = normalized.replace("postgresql://", "postgresql+psycopg://", 1)

    parts = urlsplit(normalized)
    if parts.scheme.startswith("postgresql"):
        try:
            parsed_port = parts.port
        except ValueError as exc:
            raise RuntimeError(
                "DATABASE_URL contains an invalid PostgreSQL port. "
                "Please check the host:port section and use the exact Zeabur connection string."
            ) from exc
        if parsed_port is None and ":" in parts.netloc.rsplit("@", 1)[-1]:
            raise RuntimeError(
                "DATABASE_URL contains an invalid PostgreSQL port. "
                "Please check the host:port section and use the exact Zeabur connection string."
            )
    return normalized


def _sqlite_connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def ensure_runtime_dirs() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs()


def _resolved_database_url() -> str:
    try:
        return normalize_database_url(settings.database_url)
    except RuntimeError:
        if "pytest" in sys.modules:
            return "sqlite:///./data/test_app.db"
        raise

engine = create_engine(
    _resolved_database_url(),
    future=True,
    pool_pre_ping=True,
    connect_args=_sqlite_connect_args(_resolved_database_url()),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_compat_columns()


def get_session() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


def _ensure_sqlite_compat_columns() -> None:
    if not _resolved_database_url().startswith("sqlite"):
        return

    expected_columns = {
        "fundamental_snapshots": {
            "revenue_mom": "FLOAT",
            "eps": "FLOAT",
            "roe": "FLOAT",
            "free_cash_flow": "FLOAT",
            "pe_ratio": "FLOAT",
            "pb_ratio": "FLOAT",
            "dividend_yield": "FLOAT",
        },
        "screening_candidates": {
            "rank_position": "INTEGER",
            "symbol_name": "VARCHAR(120)",
            "take_profit_1": "FLOAT",
            "take_profit_2": "FLOAT",
            "holding_days_min": "INTEGER",
            "holding_days_max": "INTEGER",
            "risk_flags": "JSON DEFAULT '{}'",
            "value_score": "FLOAT",
            "growth_score": "FLOAT",
            "quality_score": "FLOAT",
            "valuation_score": "FLOAT",
            "catalyst_score": "FLOAT",
            "value_summary": "TEXT",
            "value_risks": "JSON DEFAULT '{}'",
        },
        "holdings": {
            "opened_date": "DATE",
            "custom_stop_loss": "FLOAT",
            "custom_target_price": "FLOAT",
            "symbol_name": "VARCHAR(120)",
            "current_trend": "VARCHAR(30)",
            "alert_status": "VARCHAR(30)",
            "last_action": "VARCHAR(30)",
            "last_confidence": "FLOAT",
            "last_monitor_reasons": "JSON DEFAULT '{}'",
            "last_monitor_evidence": "JSON DEFAULT '{}'",
            "last_monitored_at": "DATETIME",
        },
        "ai_analysis_records": {
            "cache_key": "VARCHAR(255)",
        },
    }

    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            if table_name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name in existing:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                )


init_db()
