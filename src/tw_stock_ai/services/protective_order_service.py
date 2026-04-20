import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any

from clients.okx_client import OKXClient
from config.settings import settings
from storage.position_lifecycle_store import PositionLifecycleStore
from storage.protective_order_store import ProtectiveOrderStore


class ProtectiveOrderService:
    def __init__(self) -> None:
        self.client = OKXClient()
        self.store = ProtectiveOrderStore()
        self.lifecycle = PositionLifecycleStore()
        self.instrument_cache: dict[str, dict[str, Any]] = {}

    def _algo_side(self, side: str) -> str:
        return "buy" if side == "short" else "sell"

    def _pos_side(self, side: str, account_pos_mode: str) -> str | None:
        return None if account_pos_mode in {"net", "net_mode"} and not settings.force_pos_side_in_net_mode else ("short" if side == "short" else "long")

    def _normalize_size(self, value: Any) -> float:
        try:
            size = float(value or 0.0)
        except (TypeError, ValueError):
            size = 0.0
        return size if size > 0 else 0.0

    def _load_instrument(self, inst_id: str) -> dict[str, Any]:
        cached = self.instrument_cache.get(inst_id)
        if cached:
            return cached
        for row in self.client.safe_get_instruments():
            if str(row.get("instId", "")).upper() == str(inst_id).upper():
                self.instrument_cache[inst_id] = row
                return row
        return {}

    def _quantize_down(self, value: float | None, step: str | float | None) -> float | None:
        if value is None:
            return None
        try:
            v = Decimal(str(value))
            s = Decimal(str(step or "0"))
            if s <= 0:
                return float(v)
            return float((v / s).to_integral_value(rounding=ROUND_DOWN) * s)
        except Exception:
            return float(value)

    def _sanitize_tp_sl(self, symbol: str, side: str, entry_price: float, tp: Any, sl: Any) -> tuple[float | None, float | None]:
        try:
            tp_f = float(tp) if tp not in (None, "", 0, 0.0) else None
        except Exception:
            tp_f = None
        try:
            sl_f = float(sl) if sl not in (None, "", 0, 0.0) else None
        except Exception:
            sl_f = None

        inst = self._load_instrument(symbol) if settings.protect_price_guard_enabled else {}
        tick_sz = inst.get("tickSz", "0") if inst else "0"
        tick = 0.0
        try:
            tick = float(tick_sz or 0.0)
        except Exception:
            tick = 0.0
        tp_f = self._quantize_down(tp_f, tick_sz)
        sl_f = self._quantize_down(sl_f, tick_sz)

        if entry_price <= 0:
            return tp_f, sl_f

        if side == "long":
            if tp_f is not None and tp_f <= entry_price:
                tp_f = round(entry_price + max(tick, entry_price * 0.0005), 12)
            if sl_f is not None and sl_f >= entry_price:
                sl_f = round(max(entry_price - max(tick, entry_price * 0.0005), tick if tick > 0 else 1e-12), 12)
        else:
            if tp_f is not None and tp_f >= entry_price:
                tp_f = round(max(entry_price - max(tick, entry_price * 0.0005), tick if tick > 0 else 1e-12), 12)
            if sl_f is not None and sl_f <= entry_price:
                sl_f = round(entry_price + max(tick, entry_price * 0.0005), 12)

        tp_f = self._quantize_down(tp_f, tick_sz)
        sl_f = self._quantize_down(sl_f, tick_sz)
        if tp_f is not None and tp_f <= 0:
            tp_f = None
        if sl_f is not None and sl_f <= 0:
            sl_f = None
        return tp_f, sl_f


    def _result_ok(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if str(result.get("code", "-1")) != "0":
            return False
        rows = result.get("data") or []
        if not rows:
            return True
        first = rows[0] if isinstance(rows[0], dict) else {}
        return str(first.get("sCode", "0") or "0") in {"", "0"}

    def _pending_algo_matches(self, row: Dict[str, Any], symbol: str, side: str) -> bool:
        if str(row.get("instId", "") or "") != symbol:
            return False
        algo_side = self._algo_side(side)
        row_side = str(row.get("side", "") or "").lower()
        if row_side and row_side != algo_side:
            return False
        row_pos_side = str(row.get("posSide", "") or "").lower()
        expected_pos_side = side.lower()
        if row_pos_side in {"long", "short"} and row_pos_side != expected_pos_side:
            return False
        return True

    def _cancel_stale_symbol_algos(self, symbol: str, side: str) -> Dict[str, Any]:
        if not settings.enable_live_execution:
            return {"code": "0", "msg": "skip_cancel_paper_mode", "data": []}
        payload = self.client.safe_get_pending_algo_orders("conditional")
        rows = payload.get("data") or []
        cancel_items = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not self._pending_algo_matches(row, symbol, side):
                continue
            algo_id = str(row.get("algoId", "") or "")
            inst_id = str(row.get("instId", "") or symbol)
            if algo_id:
                cancel_items.append({"instId": inst_id, "algoId": algo_id})
        if not cancel_items:
            return {"code": "0", "msg": "no_stale_algo_orders", "data": []}
        return self.client.safe_cancel_algo_orders(cancel_items)

    def register(self, execution_record: Dict[str, Any], account_pos_mode: str) -> Dict[str, Any]:
        symbol = execution_record["symbol"]
        side = execution_record["side"]
        if settings.skip_protective_if_entry_failed and not execution_record.get("order_success", True):
            result = {"code": "0", "msg": "skip_protective_entry_failed", "data": []}
            record = {"symbol": symbol, "mode": "live" if settings.enable_live_execution else "paper", "action": "register", "tp": None, "sl": None, "size": 0.0, "result": result, "pos_side_used": None, "timestamp": time.time()}
            self.store.append(record)
            return record

        size = self._normalize_size(execution_record.get("final_size", execution_record.get("desired_size", 0.0)))
        entry_price = float(execution_record.get("entry_price", 0.0) or 0.0)
        raw_tp = execution_record.get("tp_sl", {}).get("take_profit_price")
        raw_sl = execution_record.get("tp_sl", {}).get("stop_loss_price")
        tp, sl = self._sanitize_tp_sl(symbol, side, entry_price, raw_tp, raw_sl)
        algo_side = self._algo_side(side)
        pos_side = self._pos_side(side, account_pos_mode)

        if size <= 0:
            result = {"code": "-1", "msg": "invalid_protective_order_size", "data": []}
        elif tp is None and sl is None:
            result = {"code": "0", "msg": "skip_protective_no_valid_trigger", "data": []}
        else:
            cancel_result = self._cancel_stale_symbol_algos(symbol, side) if settings.enable_protective_orders else {"code": "0", "msg": "skip_cancel_disabled", "data": []}
            result = self.client.safe_place_algo_tp_sl(inst_id=symbol, side=algo_side, pos_side=pos_side, tp_trigger_px=tp, sl_trigger_px=sl, size=size, margin_mode=settings.td_mode, fallback_pos_side=side) if settings.enable_live_execution and settings.enable_protective_orders else {"code": "0", "data": [{"algoId": f"paper-protect-{symbol}"}]}
            size = self._normalize_size(result.get("normalized_size", size))

        record = {"symbol": symbol, "mode": "live" if settings.enable_live_execution else "paper", "action": "register", "tp": tp, "sl": sl, "size": size, "result": result, "cancel_result": cancel_result if 'cancel_result' in locals() else {"code": "0", "msg": "skip_cancel"}, "pos_side_used": pos_side, "timestamp": time.time()}
        self.lifecycle.mark_refresh(symbol, side, "register")
        self.store.append(record)
        return record

    def refresh(self, symbol: str, side: str, size: float, tp: float | None, sl: float | None, account_pos_mode: str, reason: str, entry_price: float | None = None) -> Dict[str, Any]:
        normalized_size = self._normalize_size(size)
        entry_ref = float(entry_price or 0.0)
        tp, sl = self._sanitize_tp_sl(symbol, side, entry_ref, tp, sl)
        algo_side = self._algo_side(side)
        pos_side = self._pos_side(side, account_pos_mode)

        if normalized_size <= 0:
            result = {"code": "-1", "msg": "invalid_protective_order_size", "data": []}
        elif tp is None and sl is None:
            result = {"code": "0", "msg": "skip_protective_no_valid_trigger", "data": []}
        else:
            cancel_result = self._cancel_stale_symbol_algos(symbol, side) if settings.enable_protective_orders else {"code": "0", "msg": "skip_cancel_disabled", "data": []}
            result = self.client.safe_place_algo_tp_sl(inst_id=symbol, side=algo_side, pos_side=pos_side, tp_trigger_px=tp, sl_trigger_px=sl, size=normalized_size, margin_mode=settings.td_mode, fallback_pos_side=side) if settings.enable_live_execution and settings.enable_protective_orders else {"code": "0", "data": [{"algoId": f"paper-refresh-{symbol}"}]}
            normalized_size = self._normalize_size(result.get("normalized_size", normalized_size))

        record = {"symbol": symbol, "mode": "live" if settings.enable_live_execution else "paper", "action": "refresh", "reason": reason, "tp": tp, "sl": sl, "size": normalized_size, "result": result, "cancel_result": cancel_result if 'cancel_result' in locals() else {"code": "0", "msg": "skip_cancel"}, "pos_side_used": pos_side, "timestamp": time.time()}
        self.lifecycle.mark_refresh(symbol, side, reason)
        self.store.append(record)
        return record

    def clear_symbol_pending_algos(self, symbol: str, side: str) -> Dict[str, Any]:
        result = self._cancel_stale_symbol_algos(symbol, side)
        record = {
            "symbol": symbol,
            "mode": "live" if settings.enable_live_execution else "paper",
            "action": "clear_pending_algos",
            "result": result,
            "pos_side_used": self._pos_side(side, self.client.safe_get_pos_mode()),
            "timestamp": time.time(),
        }
        self.store.append(record)
        return record
