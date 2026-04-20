from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.services.app_settings import build_effective_settings


@dataclass(frozen=True)
class FeatureFlag:
    name: str
    setting_key: str
    description: str


FEATURE_FLAGS: list[FeatureFlag] = [
    FeatureFlag("cost_guardrails", "feature_cost_guardrails_enabled", "啟用成本防護、限流與白名單"),
    FeatureFlag("daily_report", "feature_daily_report_enabled", "啟用每日報表產生"),
    FeatureFlag("discord_notifications", "feature_discord_notifications_enabled", "啟用 Discord 推播"),
    FeatureFlag("candidate_ai_analysis", "feature_candidate_ai_analysis_enabled", "允許候選股 AI 說明"),
    FeatureFlag("holding_ai_analysis", "feature_holding_ai_analysis_enabled", "允許持股 AI 深度分析"),
    FeatureFlag("news_fetch", "feature_news_fetch_enabled", "允許抓取新聞資料"),
    FeatureFlag("cost_dashboard", "feature_cost_dashboard_enabled", "顯示成本估算面板"),
]


class FeatureFlagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def is_enabled(self, flag_name: str, session: Session | None = None) -> bool:
        effective = build_effective_settings(session) if session is not None else self.settings
        mapping = {item.name: item.setting_key for item in FEATURE_FLAGS}
        setting_key = mapping.get(flag_name)
        if setting_key is None:
            return False
        return bool(getattr(effective, setting_key))

    def describe(self, session: Session | None = None) -> list[dict]:
        effective = build_effective_settings(session) if session is not None else self.settings
        return [
            {
                "name": flag.name,
                "setting_key": flag.setting_key,
                "enabled": bool(getattr(effective, flag.setting_key)),
                "description": flag.description,
            }
            for flag in FEATURE_FLAGS
        ]
