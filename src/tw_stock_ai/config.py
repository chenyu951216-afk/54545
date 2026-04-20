from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "tw-stock-ai"
    app_env: str = "development"
    port: int = 8080
    database_url: str = "sqlite:///./data/app.db"
    price_data_provider: str = "unavailable"
    volume_data_provider: str = "unavailable"
    news_data_provider: str = "hybrid_taiwan_market_news"
    revenue_data_provider: str = "unavailable"
    fundamentals_data_provider: str = "unavailable"
    market_calendar_provider: str = "unavailable"
    fugle_api_key: str | None = None
    fugle_base_url: str = "https://api.fugle.tw/marketdata/v1.0/stock"
    fugle_timeout_seconds: int = 20
    mops_timeout_seconds: int = 30
    mops_listed_monthly_revenue_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap05_L.csv"
    mops_listed_company_profile_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
    mops_listed_daily_info_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap04_L.csv"
    mops_listed_eps_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap14_L.csv"
    mops_listed_income_statement_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap06_L_ci.csv"
    mops_listed_balance_sheet_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap07_L_ci.csv"
    mops_otc_monthly_revenue_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap05_O.csv"
    mops_otc_company_profile_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
    mops_otc_daily_info_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap04_O.csv"
    mops_otc_eps_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap14_O.csv"
    mops_otc_income_statement_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap06_O_ci.csv"
    mops_otc_balance_sheet_url: str = "https://mopsfin.twse.com.tw/opendata/t187ap07_O_ci.csv"
    twse_valuation_url: str = "https://www.twse.com.tw/exchangeReport/BWIBBU_ALL?response=open_data"
    tpex_valuation_url: str = "https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&o=csv"
    twse_holiday_schedule_url: str = "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&date={year}"
    finmind_api_token: str | None = None
    finmind_api_base_url: str = "https://api.finmindtrade.com/api/v4/data"
    finmind_timeout_seconds: int = 20
    finmind_news_dataset: str = "TaiwanStockNews"
    finmind_cash_flow_dataset: str = "TaiwanStockCashFlowsStatement"
    premium_news_provider_enabled: bool = True
    premium_news_window_start_hour: int = 7
    premium_news_window_end_hour: int = 9
    log_level: str = "INFO"
    log_file_max_bytes: int = 1048576
    log_backup_count: int = 5
    discord_webhook_url: str | None = None
    discord_enabled: bool = True
    discord_timeout_seconds: int = 10
    discord_retry_attempts: int = 3
    discord_retry_backoff_seconds: float = 1.5
    discord_daily_report_top_n: int = 5
    discord_reason_max_length: int = 120
    discord_risk_max_length: int = 120
    feature_cost_guardrails_enabled: bool = True
    feature_daily_report_enabled: bool = True
    feature_discord_notifications_enabled: bool = True
    feature_candidate_ai_analysis_enabled: bool = True
    feature_holding_ai_analysis_enabled: bool = True
    feature_news_fetch_enabled: bool = True
    feature_cost_dashboard_enabled: bool = True
    enable_scheduler: bool = False
    scheduler_timezone: str = "Asia/Taipei"
    startup_bootstrap_enabled: bool = True
    startup_bootstrap_force_refresh: bool = False
    prewarm_hour: int = 7
    prewarm_minute: int = 0
    screening_hour: int = 8
    screening_minute: int = 0
    screening_weekdays: str = "mon,tue,wed,thu,fri"
    screening_top_n: int = 5
    min_price_bars_for_screening: int = 120
    short_term_holding_days_min: int = 3
    short_term_holding_days_max: int = 10
    universe_min_close_price: float = 10.0
    universe_max_close_price: float = 3000.0
    universe_min_average_volume_20d: int = 200000
    universe_min_average_turnover_20d: float = 50000000.0
    universe_exclude_stagnant_range_ratio_20d: float = 0.03
    risk_min_reward_risk_ratio: float = 1.5
    risk_stop_atr_multiplier: float = 1.2
    risk_entry_buffer_atr_multiplier: float = 0.15
    risk_take_profit1_rr: float = 1.8
    risk_take_profit2_rr: float = 2.8
    indicator_rsi_period: int = 14
    indicator_adx_period: int = 14
    indicator_atr_period: int = 14
    indicator_macd_fast_period: int = 12
    indicator_macd_slow_period: int = 26
    indicator_macd_signal_period: int = 9
    indicator_ema_fast_period: int = 20
    indicator_ema_slow_period: int = 60
    volume_surge_ratio_threshold: float = 1.5
    breakout_lookback_days: int = 20
    breakout_buffer_pct: float = 0.01
    consolidation_lookback_days: int = 10
    consolidation_max_range_ratio: float = 0.08
    adx_trend_threshold: float = 20.0
    scoring_weight_trend: float = 0.24
    scoring_weight_momentum: float = 0.18
    scoring_weight_volume: float = 0.16
    scoring_weight_pattern: float = 0.18
    scoring_weight_strength: float = 0.12
    scoring_weight_risk: float = 0.12
    treasure_weight_growth: float = 0.28
    treasure_weight_quality: float = 0.25
    treasure_weight_valuation: float = 0.22
    treasure_weight_catalyst: float = 0.15
    treasure_weight_stability: float = 0.10
    treasure_min_required_factors: int = 3
    treasure_revenue_yoy_good: float = 15.0
    treasure_revenue_mom_good: float = 5.0
    treasure_eps_good: float = 5.0
    treasure_roe_good: float = 15.0
    treasure_gross_margin_good: float = 30.0
    treasure_operating_margin_good: float = 12.0
    treasure_free_cash_flow_good: float = 0.0
    treasure_debt_ratio_good: float = 50.0
    treasure_pe_good: float = 18.0
    treasure_pb_good: float = 2.0
    treasure_dividend_yield_good: float = 3.0
    treasure_news_lookback_days: int = 30
    treasure_recent_news_limit: int = 10
    treasure_catalyst_keywords: str = "AI,擴產,新廠,接單,新品,法說,資料中心,伺服器,CoWoS,車用,邊緣運算,資本支出"
    ai_enabled: bool = False
    ai_provider: str = "fallback"
    ai_model: str = "fallback-v1"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1/responses"
    openai_organization: str | None = None
    openai_project: str | None = None
    ai_top_n_candidates: int = 5
    ai_max_input_chars: int = 6000
    ai_max_output_tokens: int = 400
    ai_timeout_seconds: int = 20
    ai_cache_ttl_hours: int = 24
    ai_allowed_candidate_prompt_names: str = (
        "candidate_news_summary,candidate_financial_highlights,candidate_selection_reason,candidate_risk_summary"
    )
    ai_allowed_holding_prompt_names: str = "holding_trend_review"
    ai_candidate_symbol_allowlist: str = ""
    ai_monthly_budget_twd: float = 1000.0
    overall_monthly_budget_twd: float = 1000.0
    ai_estimated_input_cost_per_1k_tokens_twd: float = 0.02
    ai_estimated_output_cost_per_1k_tokens_twd: float = 0.08
    ai_fallback_enabled: bool = True
    news_analysis_enabled: bool = True
    estimated_external_api_cost_per_call_twd: float = 0.0
    estimated_notification_cost_per_send_twd: float = 0.0
    api_rate_limit_window_minutes: int = 60
    rate_limit_screening_runs_per_window: int = 6
    rate_limit_discord_reports_per_window: int = 6
    rate_limit_candidate_ai_calls_per_window: int = 20
    rate_limit_holding_ai_calls_per_window: int = 20
    rate_limit_data_refresh_per_window: int = 12
    refresh_price_bootstrap_days: int = 220
    refresh_overlap_days: int = 7
    refresh_calendar_backfill_days: int = 30
    refresh_calendar_forward_days: int = 370
    refresh_price_cache_ttl_seconds: int = 3600
    refresh_volume_cache_ttl_seconds: int = 3600
    refresh_news_cache_ttl_seconds: int = 3600
    refresh_revenue_cache_ttl_seconds: int = 21600
    refresh_fundamentals_cache_ttl_seconds: int = 21600
    refresh_calendar_cache_ttl_seconds: int = 21600
    startup_check_recent_days: int = 7
    news_fetch_max_symbols_per_run: int = 20
    candidate_max_symbols_for_news_fetch: int = 10
    holding_support_lookback_days: int = 20
    holding_volume_anomaly_ratio: float = 1.8
    holding_trend_rsi_weak_threshold: float = 45.0
    holding_distribution_drop_pct: float = -0.03
    holding_exit_confidence_base: float = 0.5
    holding_reduce_confidence_base: float = 0.6
    holding_exit_now_confidence_base: float = 0.8
    holding_negative_news_keywords: str = "下修,衰退,虧損,調降,砍單,裁員,違約,停工,延後,事故,調查,敗訴,庫存過高"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
