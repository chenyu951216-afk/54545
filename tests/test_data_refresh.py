from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.adapters.base import (
    AdapterFetchRequest,
    AdapterFetchResult,
    FundamentalsDataAdapter,
    NewsDataAdapter,
    PriceDataAdapter,
    VolumeDataAdapter,
)
from tw_stock_ai.models import Base, DataRefreshItem, DailyVolume, PriceBar, RawDataRecord
from tw_stock_ai.services.adapter_registry import AdapterRegistry
from tw_stock_ai.services.data_refresh import DataRefreshCoordinator, get_latest_refresh_run


class FakePriceAdapter(PriceDataAdapter):
    adapter_name = "fake_price"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        cleaned = [
            {
                "symbol": "2330",
                "trade_date": date(2026, 4, 17),
                "open": 850.0,
                "high": 860.0,
                "low": 845.0,
                "close": 858.0,
                "volume": 123456,
                "source_name": self.adapter_name,
                "source_url": "https://example.com/price",
                "raw_payload": {"close": 858.0},
            }
        ]
        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready",
            fetched_at=fetched_at,
            raw_items=[{"record_key": "2330:2026-04-17", "source_url": "https://example.com/price", "close": 858.0}],
            cleaned_items=cleaned,
        )


class FakeUnavailableNewsAdapter(NewsDataAdapter):
    adapter_name = "fake_news_unavailable"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("news_temporarily_unavailable", request=request)


class FakeErrorVolumeAdapter(VolumeDataAdapter):
    adapter_name = "fake_volume_error"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        raise RuntimeError("volume adapter exploded")


class FakeFundamentalsAdapter(FundamentalsDataAdapter):
    adapter_name = "fake_fundamentals"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        cleaned = [
            {
                "symbol": "2330",
                "snapshot_date": date(2026, 3, 31),
                "source_name": self.adapter_name,
                "source_url": "https://example.com/fundamentals",
                "revenue_yoy": 20.0,
                "gross_margin": 55.0,
                "operating_margin": 40.0,
                "debt_ratio": 20.0,
                "raw_payload": {"gm": 55.0},
            }
        ]
        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready",
            fetched_at=fetched_at,
            raw_items=[{"record_key": "2330:2026-03-31", "source_url": "https://example.com/fundamentals"}],
            cleaned_items=cleaned,
        )


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def test_data_refresh_contract_supports_ready_unavailable_and_failed() -> None:
    with make_session() as session:
        registry = AdapterRegistry()
        registry.register_many(
            {
                "price": FakePriceAdapter(),
                "news": FakeUnavailableNewsAdapter(),
                "volume": FakeErrorVolumeAdapter(),
                "fundamentals": FakeFundamentalsAdapter(),
            }
        )
        coordinator = DataRefreshCoordinator(registry=registry)
        run = coordinator.refresh_all(session, trigger_source="test")

        statuses = {item.dataset: item.status for item in run.items}
        assert statuses["price"] == "ready"
        assert statuses["news"] == "unavailable"
        assert statuses["volume"] == "failed"
        assert statuses["fundamentals"] == "ready"

        assert session.scalar(select(PriceBar).where(PriceBar.symbol == "2330")) is not None
        assert session.scalar(select(RawDataRecord)) is not None


def test_data_refresh_uses_cache_on_second_run() -> None:
    with make_session() as session:
        registry = AdapterRegistry()
        registry.register("price", FakePriceAdapter())
        coordinator = DataRefreshCoordinator(registry=registry)
        request = {"price": AdapterFetchRequest(symbols=["2330"], cache_ttl_seconds=3600)}

        first_run = coordinator.refresh_all(session, requests=request, trigger_source="test")
        second_run = coordinator.refresh_all(session, requests=request, trigger_source="test")

        first_price = next(item for item in first_run.items if item.dataset == "price")
        second_price = next(item for item in second_run.items if item.dataset == "price")

        assert first_price.from_cache is False
        assert second_price.from_cache is True


def test_failed_refresh_item_keeps_error_metadata() -> None:
    with make_session() as session:
        registry = AdapterRegistry()
        registry.register("volume", FakeErrorVolumeAdapter())
        coordinator = DataRefreshCoordinator(registry=registry)
        run = coordinator.refresh_all(session, trigger_source="test")

        item = session.scalar(
            select(DataRefreshItem).where(DataRefreshItem.dataset == "volume")
        )
        assert item is not None
        assert item.status == "failed"
        assert "volume adapter exploded" in item.metadata_json["errors"][0]


def test_default_refresh_requests_bootstrap_missing_market_data() -> None:
    with make_session() as session:
        coordinator = DataRefreshCoordinator()

        requests = coordinator.build_default_requests(session)

        assert requests["price"].market_code == "ALL"
        assert requests["volume"].market_code == "ALL"
        assert requests["price"].end_date == date.today()
        assert requests["volume"].end_date == date.today()
        assert requests["price"].start_date == date.today() - timedelta(days=coordinator.settings.refresh_price_bootstrap_days)
        assert requests["news"].start_date == date.today() - timedelta(days=coordinator.settings.treasure_news_lookback_days)


def test_default_refresh_requests_use_incremental_overlap_when_market_data_exists() -> None:
    with make_session() as session:
        coordinator = DataRefreshCoordinator()
        latest_date = date.today() - timedelta(days=1)
        session.add(
            PriceBar(
                symbol="2330",
                trade_date=latest_date,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000,
                source_name="test",
                source_url="https://example.com/price",
                fetched_at=datetime.now(timezone.utc),
                raw_payload={},
            )
        )
        session.add(
            DailyVolume(
                symbol="2330",
                trade_date=latest_date,
                volume=1000,
                turnover_value=100000.0,
                source_name="test",
                source_url="https://example.com/volume",
                fetched_at=datetime.now(timezone.utc),
                raw_payload={},
            )
        )
        session.commit()

        requests = coordinator.build_default_requests(session)

        expected_start = latest_date - timedelta(days=coordinator.settings.refresh_overlap_days)
        assert requests["price"].start_date == expected_start
        assert requests["volume"].start_date == expected_start


def test_get_latest_refresh_run_returns_items() -> None:
    with make_session() as session:
        registry = AdapterRegistry()
        registry.register("price", FakePriceAdapter())
        coordinator = DataRefreshCoordinator(registry=registry)
        coordinator.refresh_all(session, trigger_source="test")

        run, items = get_latest_refresh_run(session)

        assert run is not None
        assert run.trigger_source == "test"
        assert any(item.dataset == "price" for item in items)


def test_refresh_all_can_target_only_selected_datasets() -> None:
    with make_session() as session:
        registry = AdapterRegistry()
        registry.register_many(
            {
                "price": FakePriceAdapter(),
                "news": FakeUnavailableNewsAdapter(),
            }
        )
        coordinator = DataRefreshCoordinator(registry=registry)

        run = coordinator.refresh_all(
            session,
            trigger_source="targeted",
            dataset_names=["news"],
        )

        assert [item.dataset for item in run.items] == ["news"]
