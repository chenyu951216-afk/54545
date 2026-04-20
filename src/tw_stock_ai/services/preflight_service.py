from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

from clients.okx_client import OKXClient
from config.settings import settings


class LivePreflightService:
    def __init__(self) -> None:
        self.client = OKXClient()
        self.instrument_cache: dict[str, Dict[str, Any]] = {}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, "", "None"):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def load_instrument(self, inst_id: str) -> Optional[Dict[str, Any]]:
        if not inst_id:
            return None
        if inst_id in self.instrument_cache:
            return self.instrument_cache[inst_id]
        for row in self.client.safe_get_instruments():
            if row.get("instId") == inst_id:
                self.instrument_cache[inst_id] = row
                return row
        return None

    def quantize_down(self, value: float, step: str) -> float:
        if not step or Decimal(str(step)) == 0:
            return float(value)
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        quantized = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return float(quantized)

    def _extract_max_avail(self, payload: Dict[str, Any]) -> float | None:
        for row in payload.get("data", []):
            try:
                return float(row.get("availBuy") or row.get("availSell") or 0.0)
            except (TypeError, ValueError):
                continue
        return None

    def _max_avail_with_fallback(self, inst_id: str) -> Dict[str, Any]:
        payload = self.client.safe_get_max_avail_size(inst_id, settings.td_mode)
        max_avail = self._extract_max_avail(payload)
        if max_avail is not None:
            return {"supported": True, "max_avail": max_avail, "reason": "ok"}
        msg = str(payload.get("msg", "")).lower()
        code = str(payload.get("code", ""))
        if code == "51010" or "current account mode" in msg:
            return {"supported": False, "max_avail": None, "reason": "max_avail_unsupported_for_account_mode"}
        return {"supported": False, "max_avail": None, "reason": "max_avail_unavailable"}

    def _convert_base_size_to_order_size(self, instrument: Dict[str, Any], desired_size: float) -> float:
        inst_type = str(instrument.get("instType") or settings.instrument_type or "").upper()
        if inst_type not in {"SWAP", "FUTURES"}:
            return desired_size
        ct_val = self._safe_float(instrument.get("ctVal"), 0.0)
        if ct_val <= 0:
            return desired_size
        ct_mult = self._safe_float(instrument.get("ctMult"), 1.0)
        if ct_mult <= 0:
            ct_mult = 1.0
        contract_unit = ct_val * ct_mult
        if contract_unit <= 0:
            return desired_size
        return desired_size / contract_unit

    def _min_margin_required(self, leverage: int) -> float:
        lev = max(int(leverage or 1), 1)
        return max(settings.min_margin_floor_usdt, settings.min_order_notional_usdt / lev)

    def _effective_margin_floor(self, equity: float) -> float:
        equity_floor = max(equity, 0.0) * settings.effective_learning_min_margin_ratio
        leverage_anchor = max(int(settings.default_leverage_max or 1), 1)
        micro_cap = max(settings.min_margin_floor_usdt, (settings.min_order_notional_usdt / leverage_anchor) * 1.35)
        return max(settings.min_margin_floor_usdt, min(equity_floor, micro_cap))

    def preflight(self, inst_id: str, desired_size: float, desired_price: float | None) -> Dict[str, Any]:
        instrument = self.load_instrument(inst_id)
        if not instrument:
            return {"ok": False, "reason": "instrument_not_found"}

        lot_sz = instrument.get("lotSz", "1")
        min_sz = instrument.get("minSz", "1")
        tick_sz = instrument.get("tickSz", "0.1")

        raw_order_size = self._convert_base_size_to_order_size(instrument, desired_size)
        size = self.quantize_down(raw_order_size, lot_sz)
        min_sz_float = self._safe_float(min_sz, 1.0)
        if size < min_sz_float:
            size = min_sz_float

        price = self.quantize_down(desired_price, tick_sz) if desired_price is not None else None
        max_avail = None
        max_avail_reason = "not_checked"
        if settings.require_max_avail_check:
            max_row = self._max_avail_with_fallback(inst_id)
            max_avail = max_row.get("max_avail")
            max_avail_reason = max_row.get("reason", "unknown")
        if max_avail is not None and max_avail > 0:
            size = min(size, max_avail)
            size = self.quantize_down(size, lot_sz)

        if size <= 0:
            return {"ok": False, "reason": "size_after_preflight_invalid", "max_avail_reason": max_avail_reason}

        return {
            "ok": True,
            "inst_id": inst_id,
            "lot_sz": lot_sz,
            "min_sz": min_sz,
            "tick_sz": tick_sz,
            "ct_val": instrument.get("ctVal"),
            "ct_mult": instrument.get("ctMult"),
            "raw_order_size": raw_order_size,
            "final_size": size,
            "final_price": price,
            "max_avail": max_avail,
            "max_avail_reason": max_avail_reason,
        }

    def check(self, candidate: Dict[str, Any], account_summary: Dict[str, Any], pos_mode: str) -> Dict[str, Any]:
        symbol = candidate["symbol"]
        market_snapshot = candidate.get("market_snapshot", {}) or {}
        leverage_decision = candidate.get("leverage_decision", {}) or {}
        sizing_decision = candidate.get("sizing_decision", {}) or {}

        requested_leverage = int(leverage_decision.get("leverage", settings.default_leverage_min) or settings.default_leverage_min)
        margin_pct = float(leverage_decision.get("margin_pct", settings.default_margin_pct_min) or settings.default_margin_pct_min)
        size_multiplier = float(sizing_decision.get("size_multiplier", 1.0) or 1.0)
        last_price = float(market_snapshot.get("last_price", 0.0) or 0.0)
        equity = float(account_summary.get("equity", 0.0) or 0.0)
        available_usdt = float(account_summary.get("available_equity", account_summary.get("available", equity)) or 0.0)
        usable_margin = max(0.0, available_usdt * max(0.05, min(settings.margin_reserve_ratio, 1.0)))

        leverage_cap = max(1, int(self.client.safe_get_max_leverage(symbol, settings.td_mode)))
        leverage = max(1, min(requested_leverage, leverage_cap))
        if settings.effective_learning_force_max_leverage:
            leverage = leverage_cap

        min_margin_required = self._min_margin_required(leverage)
        effective_margin_floor = self._effective_margin_floor(equity)
        desired_margin = usable_margin * margin_pct * size_multiplier
        desired_margin = max(desired_margin, min_margin_required)
        if not settings.enable_exploration_live_orders:
            desired_margin = max(desired_margin, effective_margin_floor)
        desired_margin = min(desired_margin, usable_margin)

        if usable_margin < min_margin_required:
            return {
                "blocked": True,
                "reason": "insufficient_usable_margin",
                "usable_margin": usable_margin,
                "min_margin_required": min_margin_required,
                "desired_notional": settings.min_order_notional_usdt,
            }
        if (not settings.enable_exploration_live_orders) and desired_margin + 1e-9 < effective_margin_floor:
            return {
                "blocked": True,
                "reason": "effective_learning_margin_floor_not_met",
                "usable_margin": usable_margin,
                "effective_margin_floor": effective_margin_floor,
                "learning_tier": "exploration",
            }

        desired_notional = max(desired_margin * max(leverage, 1), settings.min_order_notional_usdt)
        desired_size = round(max(desired_notional / max(last_price, 1e-9), settings.lifecycle_min_position_size), 8)

        result = self.preflight(symbol, desired_size, last_price)
        learning_tier = "effective" if desired_margin >= effective_margin_floor and leverage >= leverage_cap else "exploration"
        if result.get("ok"):
            return {
                "blocked": False,
                "reason": result.get("max_avail_reason", "ok"),
                "final_size": result.get("final_size"),
                "final_price": result.get("final_price"),
                "raw_order_size": result.get("raw_order_size"),
                "ct_val": result.get("ct_val"),
                "ct_mult": result.get("ct_mult"),
                "max_avail": result.get("max_avail"),
                "max_avail_reason": result.get("max_avail_reason", "ok"),
                "desired_margin": desired_margin,
                "desired_notional": desired_notional,
                "usable_margin": usable_margin,
                "min_margin_required": min_margin_required,
                "effective_margin_floor": effective_margin_floor,
                "leverage_cap": leverage_cap,
                "planned_leverage": leverage,
                "learning_tier": learning_tier,
            }
        return {"blocked": True, "reason": result.get("reason", "preflight_failed"), "max_avail_reason": result.get("max_avail_reason", "unknown")}


PreflightService = LivePreflightService
