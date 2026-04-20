from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class AdapterRequestPayload(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    market_code: str = "TWSE"
    force_refresh: bool = False
    cache_ttl_seconds: int = 900
    limit: int | None = None
    metadata: dict = Field(default_factory=dict)


class DataRefreshItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset: str
    adapter_name: str
    status: str
    records_received: int
    records_cleaned: int
    records_stored: int
    from_cache: bool
    detail: str | None
    fetched_at: datetime
    metadata_json: dict


class DataRefreshRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trigger_source: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    summary: dict
    items: list[DataRefreshItemRead]


class HealthResponse(BaseModel):
    app_name: str
    env: str
    database_url: str
    scheduler_enabled: bool
    time: datetime


class AIAnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    target_type: str
    target_id: int
    symbol: str | None
    analysis_kind: str
    prompt_name: str
    provider: str
    model_name: str
    status: str
    summary: str
    details: dict
    evidence_snapshot: dict
    input_tokens: int
    output_tokens: int
    estimated_cost_twd: float
    fallback_used: bool
    generated_at: datetime


class DiscordDeliveryLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    report_run_id: int
    attempt_no: int
    status: str
    webhook_url_masked: str | None
    http_status: int | None
    response_body: str | None
    error_detail: str | None
    payload_json: dict
    sent_at: datetime


class DailyReportRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    report_kind: str
    report_date: date
    trigger_source: str
    screening_run_id: int | None
    status: str
    qualified_count: int
    top_n: int
    rendered_content: str
    payload_json: dict
    error_detail: str | None
    dispatched_at: datetime | None
    created_at: datetime
    delivery_logs: list[DiscordDeliveryLogRead] = Field(default_factory=list)


class FeatureFlagRead(BaseModel):
    name: str
    setting_key: str
    enabled: bool
    description: str


class CostSnapshotRead(BaseModel):
    monthly_budget_twd: float
    monthly_estimated_cost_twd: float
    monthly_remaining_budget_twd: float
    ai_budget_twd: float
    ai_actual_cost_twd: float
    daily_usage: dict
    monthly_usage: dict
    feature_flags: list[FeatureFlagRead]
    guardrails_enabled: bool


class EffectiveSettingsRead(BaseModel):
    schedule_preview: str
    weight_sum: float
    sections: dict


class StartupCheckItemRead(BaseModel):
    name: str
    status: str
    detail: str
    metadata: dict = Field(default_factory=dict)


class StartupHoldingCoverageRead(BaseModel):
    symbol: str
    latest_price_date: str | None = None
    status: str


class StartupCheckRead(BaseModel):
    generated_at: str
    overall_status: str
    checks: list[StartupCheckItemRead]
    holding_coverage: list[StartupHoldingCoverageRead] = Field(default_factory=list)
    providers: dict
    schedule: dict


class ImportResult(BaseModel):
    adapter: str
    records_received: int
    records_inserted: int
    records_skipped: int
    status: str
    detail: str | None = None


class HoldingCreate(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    quantity: int = Field(gt=0)
    average_cost: float = Field(gt=0)
    opened_date: date | None = None
    custom_stop_loss: float | None = None
    custom_target_price: float | None = None
    symbol_name: str | None = None
    note: str | None = None


class PositionAlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    holding_id: int
    symbol: str
    alert_type: str
    severity: str
    status: str
    message: str
    evidence: dict
    triggered_at: datetime
    resolved_at: datetime | None


class HoldingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    symbol_name: str | None = None
    quantity: int
    average_cost: float
    opened_date: date | None = None
    custom_stop_loss: float | None = None
    custom_target_price: float | None = None
    note: str | None
    created_at: datetime
    latest_close: float | None = None
    unrealized_pnl: float | None = None
    trend_status: str = "unavailable"
    exit_signal: str = "unavailable"
    action: str = "hold"
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    alert_status: str = "none"
    alerts: list[PositionAlertRead] = Field(default_factory=list)
    latest_ai_analysis: AIAnalysisRead | None = None
    evidence: dict = Field(default_factory=dict)
    ai_analyses: list[AIAnalysisRead] = Field(default_factory=list)


class ScreeningCandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    rank_position: int | None
    symbol: str
    symbol_name: str | None
    status: str
    overall_score: float | None
    sub_scores: dict
    evidence: dict
    entry_zone_low: float | None
    entry_zone_high: float | None
    stop_loss: float | None
    take_profit: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    risk_reward_ratio: float | None
    holding_days_min: int | None
    holding_days_max: int | None
    risk_flags: dict
    treasure_status: str
    treasure_score: float | None
    value_score: float | None
    growth_score: float | None
    quality_score: float | None
    valuation_score: float | None
    catalyst_score: float | None
    value_summary: str | None
    value_risks: dict
    treasure_evidence: dict
    ai_analyses: list[AIAnalysisRead] = Field(default_factory=list)


class ScreeningRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    as_of_date: date
    status: str
    universe_size: int
    notes: str | None
    created_at: datetime
    candidates: list[ScreeningCandidateRead]
