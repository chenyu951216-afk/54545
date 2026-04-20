from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.models import Base, Holding, NewsItem, PositionAlert, PriceBar
from tw_stock_ai.services.position_monitor import PositionMonitorService


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def _make_bar(index: int, *, close: float, volume: int, high_offset: float = 1.2, low_offset: float = 1.0) -> PriceBar:
    trade_date = date(2026, 1, 1) + timedelta(days=index)
    return PriceBar(
        symbol="2330",
        trade_date=trade_date,
        open=close * 0.995,
        high=close + high_offset,
        low=close - low_offset,
        close=close,
        volume=volume,
        source_name="test",
        source_url="https://example.com",
        fetched_at=datetime.now(timezone.utc),
        raw_payload={"symbol_name": "TSMC"},
    )


def test_position_monitor_flags_stop_loss_break_as_exit_now() -> None:
    with make_session() as session:
        holding = Holding(
            symbol="2330",
            quantity=1000,
            average_cost=100.0,
            opened_date=date(2026, 1, 5),
            custom_stop_loss=110.0,
            custom_target_price=140.0,
            note=None,
        )
        session.add(holding)
        session.add_all([_make_bar(i, close=120 + i, volume=600000) for i in range(119)])
        session.add(_make_bar(119, close=108.0, volume=1200000))
        session.commit()
        session.refresh(holding)

        result = PositionMonitorService().monitor_position(session, holding)

        assert result.action == "exit_now"
        assert "stop_loss_break" in result.reasons
        assert result.alert_status == "active"


def test_position_monitor_marks_take_profit_zone_as_reduce() -> None:
    with make_session() as session:
        holding = Holding(
            symbol="2330",
            quantity=1000,
            average_cost=100.0,
            opened_date=date(2026, 1, 5),
            custom_stop_loss=90.0,
            custom_target_price=130.0,
            note=None,
        )
        session.add(holding)
        session.add_all([_make_bar(i, close=100 + i, volume=700000) for i in range(119)])
        session.add(_make_bar(119, close=132.0, volume=900000))
        session.commit()
        session.refresh(holding)

        result = PositionMonitorService().monitor_position(session, holding)

        assert result.action == "reduce"
        assert "take_profit_zone" in result.reasons


def test_position_monitor_detects_negative_news_alert() -> None:
    with make_session() as session:
        holding = Holding(
            symbol="2330",
            quantity=1000,
            average_cost=100.0,
            opened_date=date(2026, 1, 5),
            custom_stop_loss=85.0,
            custom_target_price=150.0,
            note=None,
        )
        session.add(holding)
        session.add_all([_make_bar(i, close=100 + (i * 0.8), volume=650000) for i in range(120)])
        session.add(
            NewsItem(
                symbol="2330",
                title="公司下修展望並傳出砍單",
                source_name="test",
                source_url="https://example.com/news",
                published_at=datetime.now(timezone.utc),
                raw_payload={},
            )
        )
        session.commit()
        session.refresh(holding)

        result = PositionMonitorService().monitor_position(session, holding)
        alerts = session.scalars(select(PositionAlert).where(PositionAlert.holding_id == holding.id)).all()

        assert "negative_news" in result.reasons
        assert any(alert.alert_type == "negative_news" for alert in alerts)
