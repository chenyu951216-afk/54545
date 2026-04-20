from __future__ import annotations

from tw_stock_ai.adapters.base import BaseDataAdapter
from tw_stock_ai.adapters.fundamentals import (
    TpexMopsOtcFundamentalsAdapter,
    TwseMopsListedFundamentalsAdapter,
    TwseTpexMopsAllFundamentalsAdapter,
)
from tw_stock_ai.adapters.market_calendar import TwseHolidayCalendarAdapter
from tw_stock_ai.adapters.news import (
    FinMindTaiwanStockNewsAdapter,
    HybridTaiwanMarketNewsAdapter,
    MopsAllCompanyNewsAdapter,
    MopsListedCompanyNewsAdapter,
    MopsOtcCompanyNewsAdapter,
)
from tw_stock_ai.adapters.price import FugleHistoricalPriceAdapter
from tw_stock_ai.adapters.revenue import MopsAllRevenueAdapter, MopsListedRevenueAdapter, MopsOtcRevenueAdapter
from tw_stock_ai.adapters.unavailable import (
    UnavailableFundamentalsAdapter,
    UnavailableMarketCalendarAdapter,
    UnavailableNewsAdapter,
    UnavailablePriceAdapter,
    UnavailableRevenueAdapter,
    UnavailableVolumeAdapter,
)
from tw_stock_ai.adapters.volume import FugleHistoricalVolumeAdapter
from tw_stock_ai.config import Settings, get_settings


def build_default_adapters(settings: Settings) -> dict[str, BaseDataAdapter]:
    adapters: dict[str, BaseDataAdapter] = {
        "price": UnavailablePriceAdapter(),
        "volume": UnavailableVolumeAdapter(),
        "news": UnavailableNewsAdapter(),
        "revenue": UnavailableRevenueAdapter(),
        "fundamentals": UnavailableFundamentalsAdapter(),
        "market_calendar": UnavailableMarketCalendarAdapter(),
    }
    if settings.price_data_provider == "fugle":
        adapters["price"] = FugleHistoricalPriceAdapter(settings)
    if settings.volume_data_provider == "fugle":
        adapters["volume"] = FugleHistoricalVolumeAdapter(settings)
    if settings.news_data_provider == "mops_listed_daily_info":
        adapters["news"] = MopsListedCompanyNewsAdapter(settings)
    if settings.news_data_provider == "mops_otc_daily_info":
        adapters["news"] = MopsOtcCompanyNewsAdapter(settings)
    if settings.news_data_provider == "mops_all_daily_info":
        adapters["news"] = MopsAllCompanyNewsAdapter(settings)
    if settings.news_data_provider == "finmind_taiwan_stock_news":
        adapters["news"] = FinMindTaiwanStockNewsAdapter(settings)
    if settings.news_data_provider == "hybrid_taiwan_market_news":
        adapters["news"] = HybridTaiwanMarketNewsAdapter(settings)
    if settings.revenue_data_provider == "mops_listed_monthly_revenue":
        adapters["revenue"] = MopsListedRevenueAdapter(settings)
    if settings.revenue_data_provider == "mops_otc_monthly_revenue":
        adapters["revenue"] = MopsOtcRevenueAdapter(settings)
    if settings.revenue_data_provider == "mops_all_monthly_revenue":
        adapters["revenue"] = MopsAllRevenueAdapter(settings)
    if settings.fundamentals_data_provider == "twse_mops_listed":
        adapters["fundamentals"] = TwseMopsListedFundamentalsAdapter(settings)
    if settings.fundamentals_data_provider == "tpex_mops_otc":
        adapters["fundamentals"] = TpexMopsOtcFundamentalsAdapter(settings)
    if settings.fundamentals_data_provider == "twse_tpex_mops_all":
        adapters["fundamentals"] = TwseTpexMopsAllFundamentalsAdapter(settings)
    if settings.market_calendar_provider == "twse_holiday_schedule":
        adapters["market_calendar"] = TwseHolidayCalendarAdapter(settings)
    return adapters


class AdapterRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._adapters: dict[str, BaseDataAdapter] = build_default_adapters(self.settings)

    def register(self, dataset: str, adapter: BaseDataAdapter) -> None:
        self._adapters[dataset] = adapter

    def register_many(self, adapters: dict[str, BaseDataAdapter]) -> None:
        self._adapters.update(adapters)

    def get(self, dataset: str) -> BaseDataAdapter:
        return self._adapters[dataset]

    def as_dict(self) -> dict[str, BaseDataAdapter]:
        return dict(self._adapters)
