from __future__ import annotations

from sqlalchemy.orm import Session

from tw_stock_ai.models import Holding
from tw_stock_ai.schemas import HoldingRead
from tw_stock_ai.services.position_monitor import PositionMonitorService


def enrich_holding(session: Session, holding: Holding) -> HoldingRead:
    return PositionMonitorService().monitor_position(session, holding)
