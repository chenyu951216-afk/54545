from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.models import Base
from tw_stock_ai.services.app_settings import build_effective_settings, save_settings


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def test_app_settings_override_effective_runtime_values() -> None:
    with make_session() as session:
        changed = save_settings(
            session,
            {
                "ai_enabled": "true",
                "news_analysis_enabled": "false",
                "risk_min_reward_risk_ratio": "2.2",
                "scoring_weight_trend": "0.30",
            },
        )
        effective = build_effective_settings(session)

        assert "ai_enabled" in changed
        assert effective.ai_enabled is True
        assert effective.news_analysis_enabled is False
        assert effective.risk_min_reward_risk_ratio == 2.2
        assert effective.scoring_weight_trend == 0.30
