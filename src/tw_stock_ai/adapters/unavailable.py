from __future__ import annotations

from tw_stock_ai.adapters.base import (
    AdapterFetchRequest,
    AdapterFetchResult,
    FundamentalsDataAdapter,
    MarketCalendarAdapter,
    NewsDataAdapter,
    PriceDataAdapter,
    RevenueDataAdapter,
    VolumeDataAdapter,
)


class UnavailablePriceAdapter(PriceDataAdapter):
    adapter_name = "unavailable_price"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("price_source_not_configured", request=request)


class UnavailableVolumeAdapter(VolumeDataAdapter):
    adapter_name = "unavailable_volume"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("volume_source_not_configured", request=request)


class UnavailableNewsAdapter(NewsDataAdapter):
    adapter_name = "unavailable_news"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("news_source_not_configured", request=request)


class UnavailableRevenueAdapter(RevenueDataAdapter):
    adapter_name = "unavailable_revenue"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("revenue_source_not_configured", request=request)


class UnavailableFundamentalsAdapter(FundamentalsDataAdapter):
    adapter_name = "unavailable_fundamentals"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("fundamentals_source_not_configured", request=request)


class UnavailableMarketCalendarAdapter(MarketCalendarAdapter):
    adapter_name = "unavailable_market_calendar"

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        return self.unavailable_result("market_calendar_source_not_configured", request=request)
