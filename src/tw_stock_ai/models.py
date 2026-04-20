from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DataSource(Base, TimestampMixin):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(30), default="active", nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class AdapterCacheEntry(Base, TimestampMixin):
    __tablename__ = "adapter_cache_entries"
    __table_args__ = (
        UniqueConstraint("adapter_name", "cache_key", name="uq_adapter_cache_entry"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    adapter_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="ready")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class RawDataRecord(Base, TimestampMixin):
    __tablename__ = "raw_data_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    adapter_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    record_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(String(500))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class SecurityProfile(Base, TimestampMixin):
    __tablename__ = "security_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(120))
    market: Mapped[str] = mapped_column(String(20), nullable=False, default="TWSE")
    industry: Mapped[str | None] = mapped_column(String(80))
    source_name: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(String(500))
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class PriceBar(Base, TimestampMixin):
    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", "source_name", name="uq_price_symbol_date_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DailyVolume(Base, TimestampMixin):
    __tablename__ = "daily_volumes"
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", "source_name", name="uq_volume_symbol_date_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    turnover_value: Mapped[float | None] = mapped_column(Float)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class RevenueSnapshot(Base, TimestampMixin):
    __tablename__ = "revenue_snapshots"
    __table_args__ = (
        UniqueConstraint("symbol", "revenue_month", "source_name", name="uq_revenue_symbol_month_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    revenue_month: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    monthly_revenue: Mapped[float | None] = mapped_column(Float)
    revenue_yoy: Mapped[float | None] = mapped_column(Float)
    revenue_mom: Mapped[float | None] = mapped_column(Float)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class FundamentalSnapshot(Base, TimestampMixin):
    __tablename__ = "fundamental_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revenue_yoy: Mapped[float | None] = mapped_column(Float)
    revenue_mom: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    gross_margin: Mapped[float | None] = mapped_column(Float)
    operating_margin: Mapped[float | None] = mapped_column(Float)
    free_cash_flow: Mapped[float | None] = mapped_column(Float)
    debt_ratio: Mapped[float | None] = mapped_column(Float)
    pe_ratio: Mapped[float | None] = mapped_column(Float)
    pb_ratio: Mapped[float | None] = mapped_column(Float)
    dividend_yield: Mapped[float | None] = mapped_column(Float)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class FinancialStatementSnapshot(Base, TimestampMixin):
    __tablename__ = "financial_statement_snapshots"
    __table_args__ = (
        UniqueConstraint("symbol", "statement_date", "source_name", name="uq_financial_symbol_date_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    statement_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    period_type: Mapped[str] = mapped_column(String(30), nullable=False, default="quarterly")
    revenue: Mapped[float | None] = mapped_column(Float)
    gross_profit: Mapped[float | None] = mapped_column(Float)
    operating_income: Mapped[float | None] = mapped_column(Float)
    net_income: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class NewsItem(Base, TimestampMixin):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class MarketCalendarDay(Base, TimestampMixin):
    __tablename__ = "market_calendar_days"
    __table_args__ = (
        UniqueConstraint("market_code", "trade_date", "source_name", name="uq_market_calendar_day"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    market_code: Mapped[str] = mapped_column(String(20), nullable=False, default="TWSE", index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    is_trading_day: Mapped[bool] = mapped_column(nullable=False, default=False)
    session_type: Mapped[str | None] = mapped_column(String(30))
    holiday_name: Mapped[str | None] = mapped_column(String(100))
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DataRefreshRun(Base, TimestampMixin):
    __tablename__ = "data_refresh_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trigger_source: Mapped[str] = mapped_column(String(50), nullable=False, default="manual")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DataRefreshItem(Base, TimestampMixin):
    __tablename__ = "data_refresh_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    adapter_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    records_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_cleaned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_stored: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    from_cache: Mapped[bool] = mapped_column(nullable=False, default=False)
    detail: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class AIAnalysisRecord(Base, TimestampMixin):
    __tablename__ = "ai_analysis_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(String(20), index=True)
    analysis_kind: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    prompt_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    evidence_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_twd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fallback_used: Mapped[bool] = mapped_column(nullable=False, default=False)
    cache_key: Mapped[str | None] = mapped_column(String(255), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AppSetting(Base, TimestampMixin):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), nullable=False, default="string")
    note: Mapped[str | None] = mapped_column(Text)


class UsageEvent(Base, TimestampMixin):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    units: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    estimated_cost_twd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PositionAlert(Base, TimestampMixin):
    __tablename__ = "position_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    holding_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DailyReportRun(Base, TimestampMixin):
    __tablename__ = "daily_report_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_kind: Mapped[str] = mapped_column(String(50), nullable=False, default="discord_top_picks")
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    trigger_source: Mapped[str] = mapped_column(String(30), nullable=False, default="scheduler")
    screening_run_id: Mapped[int | None] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="prepared")
    qualified_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_n: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    rendered_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_detail: Mapped[str | None] = mapped_column(Text)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DiscordDeliveryLog(Base, TimestampMixin):
    __tablename__ = "discord_delivery_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    webhook_url_masked: Mapped[str | None] = mapped_column(String(500))
    http_status: Mapped[int | None] = mapped_column(Integer)
    response_body: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ScreeningRun(Base, TimestampMixin):
    __tablename__ = "screening_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    universe_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[str | None] = mapped_column(Text)


class ScreeningCandidate(Base, TimestampMixin):
    __tablename__ = "screening_candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(index=True, nullable=False)
    rank_position: Mapped[int | None] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="missing_data")
    overall_score: Mapped[float | None] = mapped_column(Float)
    sub_scores: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    entry_zone_low: Mapped[float | None] = mapped_column(Float)
    entry_zone_high: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    take_profit_1: Mapped[float | None] = mapped_column(Float)
    take_profit_2: Mapped[float | None] = mapped_column(Float)
    risk_reward_ratio: Mapped[float | None] = mapped_column(Float)
    holding_days_min: Mapped[int | None] = mapped_column(Integer)
    holding_days_max: Mapped[int | None] = mapped_column(Integer)
    risk_flags: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    treasure_status: Mapped[str] = mapped_column(String(30), nullable=False, default="unavailable")
    treasure_score: Mapped[float | None] = mapped_column(Float)
    value_score: Mapped[float | None] = mapped_column(Float)
    growth_score: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    valuation_score: Mapped[float | None] = mapped_column(Float)
    catalyst_score: Mapped[float | None] = mapped_column(Float)
    value_summary: Mapped[str | None] = mapped_column(Text)
    value_risks: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    treasure_evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class Holding(Base, TimestampMixin):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    average_cost: Mapped[float] = mapped_column(Float, nullable=False)
    opened_date: Mapped[date | None] = mapped_column(Date)
    custom_stop_loss: Mapped[float | None] = mapped_column(Float)
    custom_target_price: Mapped[float | None] = mapped_column(Float)
    symbol_name: Mapped[str | None] = mapped_column(String(120))
    current_trend: Mapped[str | None] = mapped_column(String(30))
    alert_status: Mapped[str | None] = mapped_column(String(30))
    last_action: Mapped[str | None] = mapped_column(String(30))
    last_confidence: Mapped[float | None] = mapped_column(Float)
    last_monitor_reasons: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    last_monitor_evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    last_monitored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
