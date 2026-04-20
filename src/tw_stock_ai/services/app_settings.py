from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.models import AppSetting


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    value_type: str
    label: str
    group: str
    description: str


SETTING_DEFINITIONS: list[SettingDefinition] = [
    SettingDefinition("discord_webhook_url", "string", "Discord Webhook", "notifications", "Discord 推播 webhook URL"),
    SettingDefinition("discord_enabled", "bool", "Discord 開關", "notifications", "控制 Discord 推播總開關"),
    SettingDefinition("feature_discord_notifications_enabled", "bool", "Discord Feature Flag", "notifications", "高成本通知功能總開關"),
    SettingDefinition("ai_enabled", "bool", "AI 開關", "analysis", "控制 AI 說明層是否啟用"),
    SettingDefinition("news_analysis_enabled", "bool", "新聞分析開關", "analysis", "控制 AI 新聞摘要與新聞解讀是否啟用"),
    SettingDefinition("feature_candidate_ai_analysis_enabled", "bool", "候選股 AI", "analysis", "控制候選股 AI 說明功能"),
    SettingDefinition("feature_holding_ai_analysis_enabled", "bool", "持股 AI", "analysis", "控制持股 AI 深度分析功能"),
    SettingDefinition("startup_bootstrap_enabled", "bool", "Startup Bootstrap", "scheduler", "worker 啟動時先執行一輪資料與分析準備"),
    SettingDefinition("prewarm_hour", "int", "預熱小時", "scheduler", "每日預熱與 AI 分析開始小時，Asia/Taipei"),
    SettingDefinition("prewarm_minute", "int", "預熱分鐘", "scheduler", "每日預熱與 AI 分析開始分鐘，Asia/Taipei"),
    SettingDefinition("screening_hour", "int", "排程小時", "scheduler", "每日排程小時，Asia/Taipei"),
    SettingDefinition("screening_minute", "int", "排程分鐘", "scheduler", "每日排程分鐘，Asia/Taipei"),
    SettingDefinition("risk_min_reward_risk_ratio", "float", "風報比門檻", "risk", "低於此風報比不得入榜"),
    SettingDefinition("feature_cost_guardrails_enabled", "bool", "成本防護", "cost", "啟用成本守門、白名單與限流"),
    SettingDefinition("feature_daily_report_enabled", "bool", "每日報表", "cost", "啟用每日報表生成"),
    SettingDefinition("feature_news_fetch_enabled", "bool", "新聞抓取", "cost", "高成本新聞抓取總開關"),
    SettingDefinition("feature_cost_dashboard_enabled", "bool", "成本面板", "cost", "啟用成本估算面板"),
    SettingDefinition("overall_monthly_budget_twd", "float", "總月成本目標", "cost", "整體月成本目標（新台幣）"),
    SettingDefinition("ai_monthly_budget_twd", "float", "AI 月成本上限", "cost", "AI 月成本上限（新台幣）"),
    SettingDefinition("ai_top_n_candidates", "int", "AI 候選股上限", "cost", "只有前 N 名候選股可呼叫 AI"),
    SettingDefinition("discord_daily_report_top_n", "int", "Discord 推播檔數", "cost", "每日推播前 N 名"),
    SettingDefinition("api_rate_limit_window_minutes", "int", "限流視窗分鐘", "cost", "API 高成本操作限流視窗"),
    SettingDefinition("rate_limit_screening_runs_per_window", "int", "選股限流", "cost", "每個視窗允許的手動選股次數"),
    SettingDefinition("rate_limit_discord_reports_per_window", "int", "推播限流", "cost", "每個視窗允許的手動推播次數"),
    SettingDefinition("rate_limit_candidate_ai_calls_per_window", "int", "候選股 AI 限流", "cost", "每個視窗允許的候選股 AI 呼叫次數"),
    SettingDefinition("rate_limit_holding_ai_calls_per_window", "int", "持股 AI 限流", "cost", "每個視窗允許的持股 AI 呼叫次數"),
    SettingDefinition("rate_limit_data_refresh_per_window", "int", "資料刷新限流", "cost", "每個視窗允許的資料刷新次數"),
    SettingDefinition("scoring_weight_trend", "float", "趨勢權重", "weights", "短線趨勢子分數權重"),
    SettingDefinition("scoring_weight_momentum", "float", "動能權重", "weights", "短線動能子分數權重"),
    SettingDefinition("scoring_weight_volume", "float", "量能權重", "weights", "短線量能子分數權重"),
    SettingDefinition("scoring_weight_pattern", "float", "型態權重", "weights", "短線型態子分數權重"),
    SettingDefinition("scoring_weight_strength", "float", "強度權重", "weights", "短線強度子分數權重"),
    SettingDefinition("scoring_weight_risk", "float", "風險權重", "weights", "短線風險子分數權重"),
]


def _definition_map() -> dict[str, SettingDefinition]:
    return {item.key: item for item in SETTING_DEFINITIONS}


def _parse_value(value: str, value_type: str) -> str | bool | int | float:
    if value_type == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    return value


def _serialize_value(value: str | bool | int | float, value_type: str) -> str:
    if value_type == "bool":
        return "true" if bool(value) else "false"
    return str(value)


def build_effective_settings(session: Session | None = None) -> Settings:
    base = get_settings()
    if session is None:
        return base

    definitions = _definition_map()
    rows = session.scalars(select(AppSetting)).all()
    overrides: dict[str, str | bool | int | float] = {}
    for row in rows:
        definition = definitions.get(row.key)
        if definition is None:
            continue
        overrides[row.key] = _parse_value(row.value, definition.value_type)
    return base.model_copy(update=overrides)


def get_settings_for_ui(session: Session) -> dict[str, dict]:
    effective = build_effective_settings(session)
    rows = {
        item.key: item
        for item in session.scalars(select(AppSetting)).all()
    }
    sections: dict[str, dict] = {}
    for definition in SETTING_DEFINITIONS:
        group = sections.setdefault(definition.group, {"title": definition.group, "items": []})
        raw_value = getattr(effective, definition.key)
        group["items"].append(
            {
                "key": definition.key,
                "label": definition.label,
                "description": definition.description,
                "value_type": definition.value_type,
                "value": raw_value,
                "stored": definition.key in rows,
            }
        )

    weight_sum = round(
        float(effective.scoring_weight_trend)
        + float(effective.scoring_weight_momentum)
        + float(effective.scoring_weight_volume)
        + float(effective.scoring_weight_pattern)
        + float(effective.scoring_weight_strength)
        + float(effective.scoring_weight_risk),
        4,
    )
    return {
        "sections": sections,
        "weight_sum": weight_sum,
        "schedule_preview": (
            f"bootstrap={'on' if effective.startup_bootstrap_enabled else 'off'} / "
            f"prewarm {int(effective.prewarm_hour):02d}:{int(effective.prewarm_minute):02d} / "
            f"push {int(effective.screening_hour):02d}:{int(effective.screening_minute):02d} Asia/Taipei"
        ),
    }


def save_settings(session: Session, values: dict[str, str]) -> list[str]:
    definitions = _definition_map()
    changed: list[str] = []
    parsed_values: dict[str, str | bool | int | float] = {}

    for key, raw in values.items():
        definition = definitions.get(key)
        if definition is None:
            continue
        parsed_values[key] = _parse_value(raw, definition.value_type)

    if "screening_hour" in parsed_values and not 0 <= int(parsed_values["screening_hour"]) <= 23:
        raise ValueError("screening_hour must be between 0 and 23")
    if "screening_minute" in parsed_values and not 0 <= int(parsed_values["screening_minute"]) <= 59:
        raise ValueError("screening_minute must be between 0 and 59")
    if "prewarm_hour" in parsed_values and not 0 <= int(parsed_values["prewarm_hour"]) <= 23:
        raise ValueError("prewarm_hour must be between 0 and 23")
    if "prewarm_minute" in parsed_values and not 0 <= int(parsed_values["prewarm_minute"]) <= 59:
        raise ValueError("prewarm_minute must be between 0 and 59")
    if "risk_min_reward_risk_ratio" in parsed_values and float(parsed_values["risk_min_reward_risk_ratio"]) <= 0:
        raise ValueError("risk_min_reward_risk_ratio must be greater than 0")
    for positive_key in (
        "overall_monthly_budget_twd",
        "ai_monthly_budget_twd",
        "ai_top_n_candidates",
        "discord_daily_report_top_n",
        "api_rate_limit_window_minutes",
        "rate_limit_screening_runs_per_window",
        "rate_limit_discord_reports_per_window",
        "rate_limit_candidate_ai_calls_per_window",
        "rate_limit_holding_ai_calls_per_window",
        "rate_limit_data_refresh_per_window",
    ):
        if positive_key in parsed_values and float(parsed_values[positive_key]) <= 0:
            raise ValueError(f"{positive_key} must be greater than 0")

    for weight_key in (
        "scoring_weight_trend",
        "scoring_weight_momentum",
        "scoring_weight_volume",
        "scoring_weight_pattern",
        "scoring_weight_strength",
        "scoring_weight_risk",
    ):
        if weight_key in parsed_values and float(parsed_values[weight_key]) < 0:
            raise ValueError(f"{weight_key} must be greater than or equal to 0")

    for key, parsed in parsed_values.items():
        definition = definitions.get(key)
        serialized = _serialize_value(parsed, definition.value_type)
        row = session.scalar(select(AppSetting).where(AppSetting.key == key))
        if row is None:
            row = AppSetting(key=key, value=serialized, value_type=definition.value_type)
            session.add(row)
            changed.append(key)
        elif row.value != serialized or row.value_type != definition.value_type:
            row.value = serialized
            row.value_type = definition.value_type
            session.add(row)
            changed.append(key)

    session.commit()
    return changed
