from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from tw_stock_ai.adapters import (
    AdapterFetchRequest,
)
from tw_stock_ai.config import get_settings
from tw_stock_ai.adapters.base import BaseDataAdapter
from tw_stock_ai.models import DataRefreshItem, DataRefreshRun, DailyVolume, PriceBar
from tw_stock_ai.schemas import DataRefreshRunRead
from tw_stock_ai.services.adapter_registry import AdapterRegistry
from tw_stock_ai.services.cache import AdapterCacheService
from tw_stock_ai.services.data_store import DataStoreService
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.logging_config import get_logger
from tw_stock_ai.services.serialization import to_jsonable
from tw_stock_ai.services.usage_tracking import UsageTracker

logger = get_logger("tw_stock_ai.data_refresh")


class DataRefreshCoordinator:
    def __init__(
        self,
        adapters: dict[str, BaseDataAdapter] | None = None,
        registry: AdapterRegistry | None = None,
        *,
        cache_service: AdapterCacheService | None = None,
        data_store: DataStoreService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.registry = registry or AdapterRegistry()
        if adapters:
            self.registry.register_many(adapters)
        self.adapters = self.registry.as_dict()
        self.cache_service = cache_service or AdapterCacheService()
        self.data_store = data_store or DataStoreService()
        self.flags = FeatureFlagService(self.settings)
        self.usage_tracker = UsageTracker()

    def build_default_requests(
        self,
        session: Session,
        *,
        force_refresh: bool = False,
    ) -> dict[str, AdapterFetchRequest]:
        today = datetime.now(ZoneInfo(self.settings.scheduler_timezone)).date()
        bootstrap_days = max(
            self.settings.refresh_price_bootstrap_days,
            self.settings.min_price_bars_for_screening + 40,
        )
        overlap_days = self.settings.refresh_overlap_days

        latest_price_date = session.scalar(select(func.max(PriceBar.trade_date)))
        latest_volume_date = session.scalar(select(func.max(DailyVolume.trade_date)))

        price_start = (
            latest_price_date - timedelta(days=overlap_days)
            if latest_price_date is not None
            else today - timedelta(days=bootstrap_days)
        )
        volume_start = (
            latest_volume_date - timedelta(days=overlap_days)
            if latest_volume_date is not None
            else today - timedelta(days=bootstrap_days)
        )

        return {
            "price": AdapterFetchRequest(
                start_date=price_start,
                end_date=today,
                market_code="ALL",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_price_cache_ttl_seconds,
            ),
            "volume": AdapterFetchRequest(
                start_date=volume_start,
                end_date=today,
                market_code="ALL",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_volume_cache_ttl_seconds,
            ),
            "news": AdapterFetchRequest(
                start_date=today - timedelta(days=self.settings.treasure_news_lookback_days),
                end_date=today,
                market_code="ALL",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_news_cache_ttl_seconds,
            ),
            "revenue": AdapterFetchRequest(
                market_code="ALL",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_revenue_cache_ttl_seconds,
            ),
            "fundamentals": AdapterFetchRequest(
                market_code="ALL",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_fundamentals_cache_ttl_seconds,
            ),
            "market_calendar": AdapterFetchRequest(
                start_date=today - timedelta(days=self.settings.refresh_calendar_backfill_days),
                end_date=today + timedelta(days=self.settings.refresh_calendar_forward_days),
                market_code="TWSE",
                force_refresh=force_refresh,
                cache_ttl_seconds=self.settings.refresh_calendar_cache_ttl_seconds,
            ),
        }

    def refresh_default(
        self,
        session: Session,
        *,
        trigger_source: str = "manual",
        force_refresh: bool = False,
    ) -> DataRefreshRunRead:
        return self.refresh_all(
            session,
            requests=self.build_default_requests(session, force_refresh=force_refresh),
            trigger_source=trigger_source,
        )

    def refresh_all(
        self,
        session: Session,
        requests: dict[str, AdapterFetchRequest] | None = None,
        *,
        trigger_source: str = "manual",
        dataset_names: list[str] | None = None,
    ) -> DataRefreshRunRead:
        requests = requests or {}
        dataset_names = dataset_names or list(self.adapters.keys())
        started_at = datetime.now(timezone.utc)
        run = DataRefreshRun(
            trigger_source=trigger_source,
            status="running",
            started_at=started_at,
            summary={},
        )
        session.add(run)
        session.flush()
        logger.info("data_refresh_started run_id=%s trigger_source=%s", run.id, trigger_source)

        items: list[DataRefreshItem] = []
        overall_status = "completed"
        summary: dict[str, dict] = {}

        for dataset in dataset_names:
            adapter = self.adapters[dataset]
            request = requests.get(dataset, AdapterFetchRequest())
            item, item_summary = self._refresh_dataset(
                session=session,
                run_id=run.id,
                adapter=adapter,
                request=request,
            )
            items.append(item)
            summary[dataset] = item_summary
            if item.status in {"failed", "unavailable"} and overall_status != "failed":
                overall_status = item.status

        run.status = overall_status
        run.completed_at = datetime.now(timezone.utc)
        run.summary = summary
        session.add_all(items)
        session.commit()
        session.refresh(run)
        logger.info("data_refresh_completed run_id=%s status=%s", run.id, run.status)
        return DataRefreshRunRead.model_validate({**run.__dict__, "items": items})

    def _refresh_dataset(
        self,
        *,
        session: Session,
        run_id: int,
        adapter: BaseDataAdapter,
        request: AdapterFetchRequest,
    ) -> tuple[DataRefreshItem, dict]:
        cache_key = request.cache_key()
        cached_entry = None if request.force_refresh else self.cache_service.get(
            session,
            adapter_name=adapter.adapter_name,
            cache_key=cache_key,
        )
        if cached_entry is not None:
            result = self.cache_service.to_result(cached_entry)
        else:
            if adapter.dataset == "news" and not self.flags.is_enabled("news_fetch", session):
                result = adapter.unavailable_result(
                    "news_fetch_feature_disabled",
                    request=request,
                    detail="feature_news_fetch_enabled=false",
                )
            else:
                try:
                    result = adapter.fetch(request)
                except Exception as exc:  # noqa: BLE001
                    result = adapter.unavailable_result(
                        "adapter_execution_failed",
                        request=request,
                        detail=str(exc),
                    )
                    result.errors.append(str(exc))
                    result.status = "failed"
            self.usage_tracker.record(
                session,
                event_type="external_api_call",
                operation=f"refresh:{adapter.dataset}",
                provider=adapter.adapter_name,
                status=result.status,
                estimated_cost_twd=self.settings.estimated_external_api_cost_per_call_twd,
                metadata={"symbols": request.symbols, "from_cache": False},
            )
            expires_at = adapter.cache_expiry(request)
            self.cache_service.set(
                session,
                adapter_name=adapter.adapter_name,
                dataset=adapter.dataset,
                cache_key=cache_key,
                result=result,
                expires_at=expires_at,
            )

        raw_stored = 0
        cleaned_stored = 0
        if not result.from_cache:
            raw_stored = self.data_store.persist_raw(session, result)
            cleaned_stored = self.data_store.persist_cleaned(session, result)
        item = DataRefreshItem(
            run_id=run_id,
            dataset=adapter.dataset,
            adapter_name=adapter.adapter_name,
            status=result.status,
            records_received=result.records_received,
            records_cleaned=result.records_cleaned,
            records_stored=cleaned_stored,
            from_cache=result.from_cache,
            detail=result.detail,
            fetched_at=result.fetched_at,
            metadata_json=to_jsonable(
                {
                "raw_records_stored": raw_stored,
                "errors": result.errors,
                "unavailable_reason": result.unavailable_reason,
                **result.metadata,
                }
            ),
        )
        session.flush()
        return item, {
            "status": item.status,
            "records_received": item.records_received,
            "records_cleaned": item.records_cleaned,
            "records_stored": item.records_stored,
            "from_cache": item.from_cache,
            "detail": item.detail,
        }


def get_latest_refresh_run(session: Session) -> tuple[DataRefreshRun | None, list[DataRefreshItem]]:
    run = session.scalar(select(DataRefreshRun).order_by(desc(DataRefreshRun.created_at), desc(DataRefreshRun.id)))
    if run is None:
        return None, []
    items = session.scalars(
        select(DataRefreshItem)
        .where(DataRefreshItem.run_id == run.id)
        .order_by(DataRefreshItem.dataset.asc(), DataRefreshItem.id.asc())
    ).all()
    return run, items


def list_recent_refresh_runs(
    session: Session,
    *,
    limit: int = 20,
) -> list[tuple[DataRefreshRun, list[DataRefreshItem]]]:
    runs = session.scalars(
        select(DataRefreshRun).order_by(desc(DataRefreshRun.created_at), desc(DataRefreshRun.id)).limit(limit)
    ).all()
    results: list[tuple[DataRefreshRun, list[DataRefreshItem]]] = []
    for run in runs:
        items = session.scalars(
            select(DataRefreshItem)
            .where(DataRefreshItem.run_id == run.id)
            .order_by(DataRefreshItem.dataset.asc(), DataRefreshItem.id.asc())
        ).all()
        results.append((run, items))
    return results
