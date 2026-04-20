from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from tw_stock_ai.adapters.base import AdapterFetchRequest, AdapterFetchResult, FundamentalsDataAdapter
from tw_stock_ai.adapters.http_utils import (
    HttpFetchError,
    build_url,
    http_get_csv_rows,
    http_get_json,
    parse_float,
    roc_date_to_date,
)
from tw_stock_ai.adapters.unavailable import UnavailableFundamentalsAdapter
from tw_stock_ai.config import Settings, get_settings


def _get_first(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _merge_results(
    *,
    adapter_name: str,
    dataset: str,
    fetched_at: datetime,
    results: list[AdapterFetchResult],
    metadata: dict | None = None,
) -> AdapterFetchResult:
    statuses = [result.status for result in results]
    if any(status == "ready" for status in statuses):
        status = "ready"
    elif any(status == "failed" for status in statuses):
        status = "failed"
    elif any(status == "unavailable" for status in statuses):
        status = "unavailable"
    else:
        status = "ready"

    raw_items: list[dict] = []
    cleaned_items: list[dict] = []
    details: list[str] = []
    errors: list[str] = []
    unavailable_reasons: list[str] = []
    for result in results:
        raw_items.extend(result.raw_items)
        cleaned_items.extend(result.cleaned_items)
        errors.extend(result.errors)
        if result.detail:
            details.append(f"{result.adapter_name}:{result.detail}")
        if result.unavailable_reason:
            unavailable_reasons.append(result.unavailable_reason)

    return AdapterFetchResult(
        adapter_name=adapter_name,
        dataset=dataset,
        status=status,
        fetched_at=fetched_at,
        raw_items=raw_items,
        cleaned_items=cleaned_items,
        detail=" | ".join(details) if details else None,
        unavailable_reason=",".join(sorted(set(unavailable_reasons))) or None,
        errors=errors,
        metadata=metadata or {},
    )


class MopsFundamentalsAdapterBase(FundamentalsDataAdapter):
    adapter_name = "mops_fundamentals"
    scope = "listed_companies"
    market = "TWSE"

    company_id_keys = ["公司代號"]
    report_date_keys = ["出表日期"]
    short_name_keys = ["公司簡稱", "公司名稱"]
    company_name_keys = ["公司名稱"]
    industry_keys = ["產業別"]
    revenue_keys = ["營業收入"]
    gross_profit_keys = ["營業毛利（毛損）淨額", "營業毛利（毛損）", "營業毛利(毛損)淨額", "營業毛利(毛損)"]
    operating_income_keys = ["營業利益（損失）", "營業利益(損失)"]
    net_income_keys = ["本期淨利（淨損）", "本期淨利(淨損)", "稅後淨利"]
    eps_keys = ["基本每股盈餘(元)", "基本每股盈餘（元）"]
    total_assets_keys = ["資產總額", "資產總計"]
    total_liabilities_keys = ["負債總額", "負債總計"]
    total_equity_keys = ["權益總額", "權益總計"]

    def __init__(self, settings: Settings | None = None, *, include_cash_flow: bool = True) -> None:
        self.settings = settings or get_settings()
        self.include_cash_flow = include_cash_flow

    @property
    def profile_url(self) -> str:
        raise NotImplementedError

    @property
    def valuation_url(self) -> str:
        raise NotImplementedError

    @property
    def eps_url(self) -> str:
        raise NotImplementedError

    @property
    def income_statement_url(self) -> str:
        raise NotImplementedError

    @property
    def balance_sheet_url(self) -> str:
        raise NotImplementedError

    def load_valuation_rows(self) -> tuple[list[dict[str, str]], list[str]]:
        return http_get_csv_rows(self.valuation_url, timeout=self.settings.mops_timeout_seconds)

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        symbol_filter = set(request.symbols)
        raw_items: list[dict] = []
        cleaned_items: list[dict] = []

        profile_rows, _ = http_get_csv_rows(self.profile_url, timeout=self.settings.mops_timeout_seconds)
        valuation_rows, _ = self.load_valuation_rows()
        eps_rows, _ = http_get_csv_rows(self.eps_url, timeout=self.settings.mops_timeout_seconds)
        income_rows, _ = http_get_csv_rows(self.income_statement_url, timeout=self.settings.mops_timeout_seconds)
        balance_rows, _ = http_get_csv_rows(self.balance_sheet_url, timeout=self.settings.mops_timeout_seconds)

        profiles_by_symbol: dict[str, dict] = {}
        for row in profile_rows:
            symbol = str(_get_first(row, self.company_id_keys) or "").strip()
            if not symbol or (symbol_filter and symbol not in symbol_filter):
                continue
            profiles_by_symbol[symbol] = row

        valuation_by_symbol: dict[str, dict] = {}
        for row in valuation_rows:
            symbol = self._extract_valuation_symbol(row)
            if not symbol or (symbol_filter and symbol not in symbol_filter):
                continue
            valuation_by_symbol[symbol] = row

        latest_eps_by_symbol = self._latest_rows(eps_rows, symbol_filter=symbol_filter)
        latest_income_by_symbol = self._latest_rows(income_rows, symbol_filter=symbol_filter)
        latest_balance_by_symbol = self._latest_rows(balance_rows, symbol_filter=symbol_filter)
        cash_flow_by_symbol = self._fetch_cash_flow_by_symbol(request, symbol_filter or set())

        all_symbols = sorted(
            set(profiles_by_symbol)
            | set(valuation_by_symbol)
            | set(latest_eps_by_symbol)
            | set(latest_income_by_symbol)
            | set(latest_balance_by_symbol)
            | set(cash_flow_by_symbol)
        )

        for symbol in all_symbols:
            profile = profiles_by_symbol.get(symbol, {})
            valuation = valuation_by_symbol.get(symbol, {})
            eps_row = latest_eps_by_symbol.get(symbol, {})
            income = latest_income_by_symbol.get(symbol, {})
            balance = latest_balance_by_symbol.get(symbol, {})
            cash_flow = cash_flow_by_symbol.get(symbol, {})
            snapshot_date = self._resolve_snapshot_date(
                self._extract_valuation_date(valuation),
                _get_first(profile, self.report_date_keys),
                _get_first(eps_row, self.report_date_keys),
                _get_first(income, self.report_date_keys),
                _get_first(balance, self.report_date_keys),
                cash_flow.get("snapshot_date"),
            ) or date.today()

            if profile:
                raw_items.append(
                    {
                        "record_key": f"profile:{self.market}:{symbol}",
                        "source_url": self.profile_url,
                        "symbol": symbol,
                        "payload": profile,
                    }
                )
                cleaned_items.append(
                    {
                        "statement_kind": "security_profile",
                        "symbol": symbol,
                        "snapshot_date": snapshot_date,
                        "symbol_name": _get_first(profile, self.short_name_keys),
                        "company_name": _get_first(profile, self.company_name_keys),
                        "industry": _get_first(profile, self.industry_keys),
                        "market": self.market,
                        "source_name": self.adapter_name,
                        "source_url": self.profile_url,
                        "raw_payload": {"provider": "mops", "scope": self.scope, **profile},
                    }
                )

            pe_ratio = self._extract_pe_ratio(valuation)
            pb_ratio = self._extract_pb_ratio(valuation)
            dividend_yield = self._extract_dividend_yield(valuation)
            eps_value = parse_float(_get_first(eps_row, self.eps_keys)) or parse_float(_get_first(income, self.eps_keys))
            revenue = parse_float(_get_first(income, self.revenue_keys))
            gross_profit = parse_float(_get_first(income, self.gross_profit_keys))
            operating_income = parse_float(_get_first(income, self.operating_income_keys))
            net_income = parse_float(_get_first(income, self.net_income_keys))
            debt_ratio = self._calculate_debt_ratio(balance)
            gross_margin = (gross_profit / revenue * 100.0) if gross_profit is not None and revenue not in (None, 0.0) else None
            operating_margin = (
                operating_income / revenue * 100.0
                if operating_income is not None and revenue not in (None, 0.0)
                else None
            )
            roe = self._calculate_roe(net_income, balance, pe_ratio=pe_ratio, pb_ratio=pb_ratio)
            free_cash_flow = cash_flow.get("free_cash_flow")

            raw_items.append(
                {
                    "record_key": f"fundamentals:{self.market}:{symbol}:{snapshot_date.isoformat()}",
                    "source_url": self.valuation_url,
                    "symbol": symbol,
                    "payload": {
                        "valuation": valuation,
                        "eps": eps_row,
                        "income": income,
                        "balance": balance,
                        "cash_flow": cash_flow.get("raw_payload"),
                    },
                }
            )
            cleaned_items.append(
                {
                    "statement_kind": "fundamentals",
                    "symbol": symbol,
                    "snapshot_date": snapshot_date,
                    "source_name": self.adapter_name,
                    "source_url": self.valuation_url,
                    "eps": eps_value,
                    "roe": roe,
                    "gross_margin": gross_margin,
                    "operating_margin": operating_margin,
                    "free_cash_flow": free_cash_flow,
                    "debt_ratio": debt_ratio,
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "dividend_yield": dividend_yield,
                    "raw_payload": {
                        "provider": "mops",
                        "scope": self.scope,
                        "derived": {
                            "debt_ratio_formula": "total_liabilities / total_assets * 100" if debt_ratio is not None else None,
                            "gross_margin_formula": "gross_profit / revenue * 100" if gross_margin is not None else None,
                            "operating_margin_formula": "operating_income / revenue * 100" if operating_margin is not None else None,
                            "roe_formula": (
                                "net_income / total_equity * 100"
                                if net_income is not None and _get_first(balance, self.total_equity_keys)
                                else "pb_ratio / pe_ratio * 100"
                                if roe is not None and pe_ratio not in (None, 0.0) and pb_ratio is not None
                                else None
                            ),
                            "free_cash_flow_formula": cash_flow.get("formula"),
                        },
                        "valuation": valuation,
                        "eps": eps_row,
                        "income": income,
                        "balance": balance,
                        "cash_flow": cash_flow.get("raw_payload"),
                    },
                }
            )
            if income or eps_row:
                cleaned_items.append(
                    {
                        "statement_kind": "financial_statement",
                        "symbol": symbol,
                        "statement_date": snapshot_date,
                        "period_type": "quarterly",
                        "revenue": revenue,
                        "gross_profit": gross_profit,
                        "operating_income": operating_income,
                        "net_income": net_income,
                        "eps": eps_value,
                        "source_name": self.adapter_name,
                        "source_url": self.income_statement_url,
                        "raw_payload": {"provider": "mops", "scope": self.scope, "income": income, "eps": eps_row},
                    }
                )

        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready",
            fetched_at=fetched_at,
            raw_items=raw_items,
            cleaned_items=cleaned_items,
            detail=None if cleaned_items else "no_fundamental_rows_loaded",
            metadata={"provider": "mops", "scope": self.scope, "market": self.market},
        )

    def _fetch_cash_flow_by_symbol(self, request: AdapterFetchRequest, symbols: set[str]) -> dict[str, dict]:
        if not self.include_cash_flow or not self.settings.finmind_api_token or not symbols:
            return {}

        start_date = request.start_date or (date.today() - timedelta(days=730))
        cash_flows: dict[str, dict] = {}
        for symbol in sorted(symbols):
            params = {
                "dataset": self.settings.finmind_cash_flow_dataset,
                "data_id": symbol,
                "start_date": start_date.isoformat(),
            }
            if request.end_date:
                params["end_date"] = request.end_date.isoformat()
            try:
                payload = http_get_json(
                    build_url(self.settings.finmind_api_base_url, params=params),
                    headers={"Authorization": f"Bearer {self.settings.finmind_api_token}"},
                    timeout=self.settings.finmind_timeout_seconds,
                )
            except HttpFetchError:
                continue

            grouped: dict[date, dict[str, float]] = {}
            raw_rows: dict[date, list[dict]] = {}
            for row in payload.get("data", []):
                row_date_text = str(row.get("date", "")).strip()
                if not row_date_text:
                    continue
                try:
                    row_date = date.fromisoformat(row_date_text[:10])
                except ValueError:
                    continue
                grouped.setdefault(row_date, {})
                raw_rows.setdefault(row_date, []).append(row)
                row_type = str(row.get("type", "")).strip()
                try:
                    numeric_value = float(row.get("value"))
                except (TypeError, ValueError):
                    continue
                grouped[row_date][row_type] = numeric_value

            if not grouped:
                continue

            snapshot_date = max(grouped)
            values = grouped[snapshot_date]
            operating_cash_flow = values.get("CashFlowsFromOperatingActivities")
            capex = values.get("PropertyAndPlantAndEquipment")
            if operating_cash_flow is None:
                continue

            free_cash_flow = operating_cash_flow - abs(capex) if capex is not None else operating_cash_flow
            cash_flows[symbol] = {
                "snapshot_date": snapshot_date.isoformat(),
                "free_cash_flow": free_cash_flow,
                "formula": (
                    "CashFlowsFromOperatingActivities - abs(PropertyAndPlantAndEquipment)"
                    if capex is not None
                    else "CashFlowsFromOperatingActivities"
                ),
                "raw_payload": {
                    "provider": "finmind",
                    "dataset": self.settings.finmind_cash_flow_dataset,
                    "rows": raw_rows.get(snapshot_date, []),
                },
            }
        return cash_flows

    def _latest_rows(self, rows: list[dict[str, str]], *, symbol_filter: set[str]) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for row in rows:
            symbol = str(_get_first(row, self.company_id_keys) or "").strip()
            if not symbol or (symbol_filter and symbol not in symbol_filter):
                continue
            current_date = roc_date_to_date(_get_first(row, self.report_date_keys))
            previous_date = roc_date_to_date(_get_first(latest.get(symbol, {}), self.report_date_keys))
            if previous_date is None or (current_date is not None and current_date >= previous_date):
                latest[symbol] = row
        return latest

    @staticmethod
    def _resolve_snapshot_date(*values: str | None) -> date | None:
        for value in values:
            if value is None:
                continue
            parsed = roc_date_to_date(value)
            if parsed is not None:
                return parsed
            try:
                parsed = date.fromisoformat(str(value)[:10])
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed
        return None

    def _calculate_debt_ratio(self, balance_row: dict[str, str]) -> float | None:
        total_assets = parse_float(_get_first(balance_row, self.total_assets_keys))
        total_liabilities = parse_float(_get_first(balance_row, self.total_liabilities_keys))
        if total_assets in (None, 0.0) or total_liabilities is None:
            return None
        return total_liabilities / total_assets * 100.0

    def _calculate_roe(
        self,
        net_income: float | None,
        balance_row: dict[str, str],
        *,
        pe_ratio: float | None,
        pb_ratio: float | None,
    ) -> float | None:
        total_equity = parse_float(_get_first(balance_row, self.total_equity_keys))
        if net_income is not None and total_equity not in (None, 0.0):
            return net_income / total_equity * 100.0
        if pe_ratio not in (None, 0.0) and pb_ratio is not None:
            return pb_ratio / pe_ratio * 100.0
        return None

    @staticmethod
    def _extract_valuation_symbol(row: dict[str, str]) -> str:
        return str(_get_first(row, ["股票代號"]) or "").strip()

    @staticmethod
    def _extract_valuation_date(row: dict[str, str]) -> str | None:
        return _get_first(row, ["日期"])

    @staticmethod
    def _extract_pe_ratio(row: dict[str, str]) -> float | None:
        return parse_float(_get_first(row, ["本益比"]))

    @staticmethod
    def _extract_pb_ratio(row: dict[str, str]) -> float | None:
        return parse_float(_get_first(row, ["股價淨值比"]))

    @staticmethod
    def _extract_dividend_yield(row: dict[str, str]) -> float | None:
        return parse_float(_get_first(row, ["殖利率(%)"]))


class TwseMopsListedFundamentalsAdapter(MopsFundamentalsAdapterBase):
    adapter_name = "twse_mops_listed_fundamentals"
    scope = "listed_companies"
    market = "TWSE"

    @property
    def profile_url(self) -> str:
        return self.settings.mops_listed_company_profile_url

    @property
    def valuation_url(self) -> str:
        return self.settings.twse_valuation_url

    @property
    def eps_url(self) -> str:
        return self.settings.mops_listed_eps_url

    @property
    def income_statement_url(self) -> str:
        return self.settings.mops_listed_income_statement_url

    @property
    def balance_sheet_url(self) -> str:
        return self.settings.mops_listed_balance_sheet_url


class TpexMopsOtcFundamentalsAdapter(MopsFundamentalsAdapterBase):
    adapter_name = "tpex_mops_otc_fundamentals"
    scope = "otc_companies"
    market = "TPEx"

    @property
    def profile_url(self) -> str:
        return self.settings.mops_otc_company_profile_url

    @property
    def valuation_url(self) -> str:
        return self.settings.tpex_valuation_url

    @property
    def eps_url(self) -> str:
        return self.settings.mops_otc_eps_url

    @property
    def income_statement_url(self) -> str:
        return self.settings.mops_otc_income_statement_url

    @property
    def balance_sheet_url(self) -> str:
        return self.settings.mops_otc_balance_sheet_url

    def load_valuation_rows(self) -> tuple[list[dict[str, str]], list[str]]:
        return http_get_csv_rows(
            self.valuation_url,
            timeout=self.settings.mops_timeout_seconds,
            skip_lines=3,
            encoding="cp950",
        )


class TwseTpexMopsAllFundamentalsAdapter(FundamentalsDataAdapter):
    adapter_name = "twse_tpex_mops_all_fundamentals"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.listed_adapter = TwseMopsListedFundamentalsAdapter(self.settings, include_cash_flow=False)
        self.otc_adapter = TpexMopsOtcFundamentalsAdapter(self.settings, include_cash_flow=False)

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        merged = _merge_results(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            fetched_at=fetched_at,
            results=[
                self.listed_adapter.fetch(request),
                self.otc_adapter.fetch(request),
            ],
            metadata={"provider": "hybrid", "scope": "listed_and_otc"},
        )
        if request.symbols and self.settings.finmind_api_token:
            cash_flow_by_symbol = TwseMopsListedFundamentalsAdapter(
                self.settings,
                include_cash_flow=True,
            )._fetch_cash_flow_by_symbol(request, set(request.symbols))
            self._apply_cash_flow(merged.cleaned_items, cash_flow_by_symbol)
        return merged

    @staticmethod
    def _apply_cash_flow(cleaned_items: list[dict], cash_flow_by_symbol: dict[str, dict]) -> None:
        for item in cleaned_items:
            if item.get("statement_kind") != "fundamentals":
                continue
            cash_flow = cash_flow_by_symbol.get(item["symbol"])
            if not cash_flow:
                continue
            item["free_cash_flow"] = cash_flow.get("free_cash_flow")
            raw_payload = item.setdefault("raw_payload", {})
            derived = raw_payload.setdefault("derived", {})
            derived["free_cash_flow_formula"] = cash_flow.get("formula")
            raw_payload["cash_flow"] = cash_flow.get("raw_payload")


__all__ = [
    "FundamentalsDataAdapter",
    "TwseMopsListedFundamentalsAdapter",
    "TpexMopsOtcFundamentalsAdapter",
    "TwseTpexMopsAllFundamentalsAdapter",
    "UnavailableFundamentalsAdapter",
]
