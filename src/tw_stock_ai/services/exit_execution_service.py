from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

from ai.adaptive_policy_store import AdaptivePolicyStore
from clients.okx_client import OKXClient
from config.settings import settings
from storage.order_store import OrderStore
from storage.position_lifecycle_store import PositionLifecycleStore
from storage.shadow_observation_store import ShadowObservationStore
from storage.trade_store import TradeStore
from services.dual_layer_ai_service import DualLayerAIService
from services.protective_order_service import ProtectiveOrderService


class ExitExecutionService:
    def __init__(self) -> None:
        self.client = OKXClient()
        self.orders = OrderStore()
        self.trades = TradeStore()
        self.policy_store = AdaptivePolicyStore()
        self.lifecycle = PositionLifecycleStore()
        self.shadow_store = ShadowObservationStore()
        self.dual_layer = DualLayerAIService()
        self.protective = ProtectiveOrderService()

    def _pos_side(self, side: str) -> str | None:
        return None if side in {"", "net"} and not settings.force_pos_side_in_net_mode else ("short" if side == "short" else "long")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_order_id(self, result: Dict[str, Any]) -> str:
        rows = result.get("data", []) if isinstance(result, dict) else []
        if rows and isinstance(rows[0], dict):
            return str(rows[0].get("ordId", "") or "")
        return ""

    def _estimate_realized_pnl_fallback(self, position: Dict[str, Any], closed_size: float) -> Dict[str, Any]:
        original_size = max(self._safe_float(position.get("size", 0.0), 0.0), settings.lifecycle_min_position_size)
        close_fraction = min(max(closed_size / max(original_size, 1e-9), 0.0), 1.0)
        upl_amount = self._safe_float(position.get("upl", 0.0), 0.0)
        upl_ratio = self._safe_float(position.get("upl_ratio", position.get("uplRatio", 0.0)), 0.0)
        entry_price = self._safe_float(position.get("entry_price", 0.0), 0.0)
        mark_price = self._safe_float(position.get("current_price", entry_price), entry_price)
        realized_amount = upl_amount * close_fraction
        return {
            "realized_pnl": realized_amount,
            "realized_pnl_gross": realized_amount,
            "realized_pnl_source": "position_upl_fallback",
            "pnl_ratio": upl_ratio * close_fraction,
            "close_price": mark_price,
            "filled_size": closed_size,
            "fill_fee": 0.0,
            "fee_usdt": 0.0,
            "fill_fee_ccy": "",
            "entry_price": entry_price,
            "close_fraction": close_fraction,
        }

    def _fetch_realized_close_snapshot(self, position: Dict[str, Any], result: Dict[str, Any], requested_size: float) -> Dict[str, Any]:
        ord_id = self._extract_order_id(result)
        inst_id = str(position.get("symbol", "") or "")
        if not settings.enable_live_execution or not ord_id or not inst_id:
            return self._estimate_realized_pnl_fallback(position, requested_size)

        detail_row: Dict[str, Any] = {}
        for _ in range(4):
            try:
                payload = self.client._request("GET", "/api/v5/trade/order", params={"instId": inst_id, "ordId": ord_id}, private=True)
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                if rows and isinstance(rows[0], dict):
                    detail_row = rows[0]
                    if detail_row.get("state") in {"filled", "partially_filled"} or detail_row.get("accFillSz") not in (None, "", "0"):
                        break
            except Exception:
                detail_row = {}
            time.sleep(0.35)

        if detail_row:
            fill_pnl = self._safe_float(detail_row.get("fillPnl"), 0.0)
            fill_fee_signed = self._safe_float(detail_row.get("fillFee"), 0.0)
            fee_usdt = abs(fill_fee_signed)
            close_price = self._safe_float(detail_row.get("avgPx") or detail_row.get("fillPx") or detail_row.get("px"), 0.0)
            filled_size = self._safe_float(detail_row.get("accFillSz") or detail_row.get("fillSz"), requested_size)
            entry_price = self._safe_float(position.get("entry_price", 0.0), 0.0)
            realized_amount = fill_pnl + fill_fee_signed
            if fill_pnl != 0.0 or fill_fee_signed != 0.0 or close_price > 0.0:
                basis = max(entry_price * max(filled_size, 0.0), 1e-9)
                pnl_ratio = realized_amount / basis if basis > 0 else 0.0
                return {
                    "realized_pnl": realized_amount,
                    "realized_pnl_gross": fill_pnl,
                    "realized_pnl_source": "okx_order_details",
                    "pnl_ratio": pnl_ratio,
                    "close_price": close_price,
                    "filled_size": filled_size,
                    "fill_fee": fill_fee_signed,
                    "fee_usdt": fee_usdt,
                    "fill_fee_ccy": str(detail_row.get("fillFeeCcy") or detail_row.get("feeCcy") or ""),
                    "entry_price": entry_price,
                    "close_fraction": min(max(filled_size / max(self._safe_float(position.get('size', 0.0), 0.0), 1e-9), 0.0), 1.0),
                    "ord_id": ord_id,
                    "trade_id": str(detail_row.get("tradeId", "") or ""),
                }
        return self._estimate_realized_pnl_fallback(position, requested_size)

    def _append_trade_record(self, position: Dict[str, Any], reason: str, size: float, management_action: str, review_area: str, execution_snapshot: Dict[str, Any], is_full_close: bool) -> None:
        policy = self.policy_store.load()
        lifecycle_state = position.get("lifecycle_state", {}) or {}
        close_time = datetime.now(timezone.utc).isoformat()
        realized_pnl = self._safe_float(execution_snapshot.get("realized_pnl"), 0.0)
        realized_pnl_gross = self._safe_float(execution_snapshot.get("realized_pnl_gross", realized_pnl), realized_pnl)
        realized_pnl_net = realized_pnl
        fee_signed = self._safe_float(execution_snapshot.get("fill_fee"), 0.0)
        fee_usdt = self._safe_float(execution_snapshot.get("fee_usdt", abs(fee_signed)), abs(fee_signed))
        filled_size = self._safe_float(execution_snapshot.get("filled_size"), size)
        entry_price = self._safe_float(execution_snapshot.get("entry_price"), self._safe_float(position.get("entry_price", 0.0), 0.0))
        close_price = self._safe_float(execution_snapshot.get("close_price"), self._safe_float(position.get("current_price", 0.0), 0.0))
        pnl_ratio = self._safe_float(execution_snapshot.get("pnl_ratio"), self._safe_float(position.get("upl_ratio", position.get("uplRatio", 0.0)), 0.0))
        leverage = float(lifecycle_state.get("applied_leverage", position.get("leverage", 0.0)) or 0.0)
        margin_used = float(lifecycle_state.get("margin_used", position.get("margin_used", position.get("effective_margin_used", 0.0))) or 0.0)
        if margin_used <= 0 and leverage > 0:
            basis = entry_price * max(filled_size, 0.0)
            margin_used = basis / leverage if leverage > 0 else 0.0
        entry_notional_usdt = entry_price * max(filled_size, 0.0)
        close_notional_usdt = close_price * max(filled_size, 0.0)
        effective_leverage_realized = entry_notional_usdt / max(margin_used, 1e-9) if margin_used > 0 else 0.0
        pnl_on_margin_pct = (realized_pnl_net / margin_used * 100.0) if margin_used > 0 else 0.0
        pnl_on_notional_bp = (realized_pnl_net / entry_notional_usdt * 10000.0) if entry_notional_usdt > 0 else 0.0
        fee_on_notional_bp = (fee_usdt / entry_notional_usdt * 10000.0) if entry_notional_usdt > 0 else 0.0
        entry_time = str(lifecycle_state.get("entry_time") or position.get("entry_time") or "")
        hold_seconds = 0.0
        if entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                close_dt = datetime.fromisoformat(close_time)
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)
                hold_seconds = max((close_dt - entry_dt).total_seconds(), 0.0)
            except Exception:
                hold_seconds = 0.0

        base_learning_tier = str(lifecycle_state.get("learning_tier") or position.get("learning_tier") or "exploration")
        margin_used_for_learning = max(self._safe_float(position.get("margin_used", lifecycle_state.get("margin_used", 0.0)), 0.0), 0.0)
        realized_abs_pnl = abs(realized_pnl_net)
        equity_snapshot = self._safe_float(position.get("equity_snapshot", 0.0), 0.0)
        leverage_cap = int(lifecycle_state.get("leverage_cap", position.get("leverage_cap", settings.default_leverage_max)) or settings.default_leverage_max)
        effective_margin_floor = max(
            settings.min_margin_floor_usdt,
            min(
                max(equity_snapshot, 0.0) * settings.effective_learning_min_margin_ratio,
                max(settings.min_margin_floor_usdt, (settings.min_order_notional_usdt / max(leverage_cap, 1)) * 1.35),
            ),
        )
        effective_pnl_floor = self.dual_layer.effective_pnl_floor(margin_used_for_learning, fee_usdt)
        qualifies_effective = (
            base_learning_tier == "effective"
            and margin_used_for_learning >= effective_margin_floor
            and realized_abs_pnl >= effective_pnl_floor
        )
        learning_tier = "effective" if qualifies_effective else "exploration"
        count_in_learning = learning_tier == "effective"

        trade_record: Dict[str, Any] = {
            "symbol": position["symbol"],
            "side": position.get("side"),
            "pnl": realized_pnl_net,
            "pnl_net": realized_pnl_net,
            "pnl_gross": realized_pnl_gross,
            "pnl_amount": realized_pnl_net,
            "pnl_ratio": pnl_ratio,
            "pnl_on_margin_pct": round(pnl_on_margin_pct, 6),
            "margin_used": round(margin_used, 8),
            "drawdown": float(position.get("max_drawdown", 0.0) or 0.0),
            "reason": reason,
            "review_area": review_area,
            "entry_confidence": float(lifecycle_state.get("entry_confidence", position.get("entry_confidence", 0.0)) or 0.0),
            "trend_bias": lifecycle_state.get("trend_bias", position.get("trend_bias")),
            "trend_stage": lifecycle_state.get("trend_stage", position.get("trend_stage", "unknown")),
            "market_regime": lifecycle_state.get("market_regime", position.get("market_regime", "unknown")),
            "market_phase": lifecycle_state.get("market_phase", position.get("market_phase", "neutral_balance")),
            "breakout_phase": lifecycle_state.get("breakout_phase", position.get("breakout_phase", "neutral")),
            "pre_breakout_score": float(lifecycle_state.get("pre_breakout_score", position.get("pre_breakout_score", 0.0)) or 0.0),
            "breakout_followthrough_score": float(lifecycle_state.get("breakout_followthrough_score", position.get("breakout_followthrough_score", 0.0)) or 0.0),
            "breakout_failure_risk": float(lifecycle_state.get("breakout_failure_risk", position.get("breakout_failure_risk", 0.0)) or 0.0),
            "rebound_probability_score": float(lifecycle_state.get("rebound_probability_score", position.get("rebound_probability_score", 0.0)) or 0.0),
            "exhaustion_risk_score": float(lifecycle_state.get("exhaustion_risk_score", position.get("exhaustion_risk_score", 0.0)) or 0.0),
            "continuation_context_score": float(lifecycle_state.get("continuation_context_score", position.get("continuation_context_score", 0.0)) or 0.0),
            "reversal_context_score": float(lifecycle_state.get("reversal_context_score", position.get("reversal_context_score", 0.0)) or 0.0),
            "entry_style_hint": str(lifecycle_state.get("entry_style_hint", position.get("entry_style_hint", "observe")) or "observe"),
            "learning_context_summary": str(lifecycle_state.get("learning_context_summary", position.get("learning_context_summary", "")) or ""),
            "market_story": str(lifecycle_state.get("market_story", position.get("market_story", "")) or ""),
            "market_basis_summary": str(lifecycle_state.get("market_basis_summary", position.get("market_basis_summary", "")) or ""),
            "market_basis_snapshot": lifecycle_state.get("market_basis_snapshot", position.get("market_basis_snapshot", {})),
            "learning_adjustments": lifecycle_state.get("learning_adjustments", position.get("learning_adjustments", {})),
            "protection_decision": position.get("protection_decision", {}),
            "management_decision": position.get("management_decision", {}),
            "exit_decision": position.get("exit_decision", {}),
            "size": size,
            "filled_size": filled_size,
            "size_multiplier": float(lifecycle_state.get("size_multiplier", position.get("size_multiplier", 1.0)) or 1.0),
            "leverage": leverage,
            "requested_leverage": float(lifecycle_state.get("requested_leverage", leverage) or leverage),
            "leverage_cap": float(lifecycle_state.get("leverage_cap", leverage) or leverage),
            "margin_pct": float(position.get("margin_pct", 0.0) or 0.0),
            "entry_price": entry_price,
            "close_price": close_price,
            "entry_notional_usdt": round(entry_notional_usdt, 8),
            "close_notional_usdt": round(close_notional_usdt, 8),
            "effective_leverage_realized": round(effective_leverage_realized, 6),
            "pnl_on_notional_bp": round(pnl_on_notional_bp, 6),
            "fee_on_notional_bp": round(fee_on_notional_bp, 6),
            "fill_fee": fee_signed,
            "fee_usdt": fee_usdt,
            "fill_fee_ccy": str(execution_snapshot.get("fill_fee_ccy", "") or ""),
            "realized_pnl_source": str(execution_snapshot.get("realized_pnl_source", "unknown") or "unknown"),
            "exit_style": policy.get("exit_style", "balanced"),
            "protection_profile": policy.get("protection_profile", "balanced"),
            "position_management_profile": policy.get("position_management_profile", "balanced"),
            "management_action": management_action,
            "protection_state": position.get("protection_state", policy.get("protection_profile", "balanced")),
            "lifecycle_stage": position.get("lifecycle_stage", "none"),
            "lifecycle_snapshot": lifecycle_state,
            "close_fraction": self._safe_float(execution_snapshot.get("close_fraction"), 1.0),
            "ord_id": str(execution_snapshot.get("ord_id", "") or ""),
            "trade_id": str(execution_snapshot.get("trade_id", "") or ""),
            "is_full_close": bool(is_full_close),
            "learning_tier": learning_tier,
            "learning_weight": settings.effective_learning_weight if count_in_learning else settings.exploration_learning_weight,
            "count_in_learning": count_in_learning and bool(is_full_close) and execution_snapshot.get("realized_pnl_source") == "okx_order_details",
            "entry_time": entry_time,
            "close_time": close_time,
            "hold_seconds": round(hold_seconds, 3),
        }
        self.trades.append(trade_record)
        self.dual_layer.record_setup_outcome(
            symbol=position["symbol"],
            side=str(position.get("side", "")),
            market_regime=str(lifecycle_state.get("market_regime", position.get("market_regime", "unknown")) or "unknown"),
            pnl_net=realized_pnl_net,
            fee=fee_usdt,
            margin=margin_used,
            count_in_learning=count_in_learning and bool(is_full_close),
        )
        if is_full_close and not count_in_learning:
            confidence_before = self._safe_float(lifecycle_state.get("entry_confidence", position.get("entry_confidence", 0.0)), 0.0)
            confidence_delta = settings.exploration_confidence_delta_win if realized_pnl_net > 0 else settings.exploration_confidence_delta_loss
            self.trades.append_exploration_sample({
                "timestamp": close_time,
                "symbol": position["symbol"],
                "setup_key": str(lifecycle_state.get("setup_key", position.get("setup_key", "")) or ""),
                "side": position.get("side", ""),
                "market_regime": lifecycle_state.get("market_regime", position.get("market_regime", "unknown")),
                "market_phase": lifecycle_state.get("market_phase", position.get("market_phase", "neutral_balance")),
                "breakout_phase": lifecycle_state.get("breakout_phase", position.get("breakout_phase", "neutral")),
                "trend_stage": lifecycle_state.get("trend_stage", position.get("trend_stage", "unknown")),
                "learning_context_summary": lifecycle_state.get("learning_context_summary", position.get("learning_context_summary", "")),
                "learning_adjustments": lifecycle_state.get("learning_adjustments", position.get("learning_adjustments", {})),
                "confidence_before": confidence_before,
                "confidence_delta": confidence_delta,
                "confidence_after": max(0.0, min(1.0, confidence_before + confidence_delta)),
                "result_label": "win" if realized_pnl_net > 0 else "loss",
                "leverage": leverage,
                "margin_used": margin_used,
                "entry_notional_usdt": round(entry_notional_usdt, 8),
                "effective_leverage_realized": round(effective_leverage_realized, 6),
                "requested_leverage": float(lifecycle_state.get("requested_leverage", leverage) or leverage),
                "leverage_cap": float(lifecycle_state.get("leverage_cap", leverage) or leverage),
                "pnl_on_margin_pct": round(pnl_on_margin_pct, 6),
                "pnl_net": realized_pnl_net,
                "fee_usdt": fee_usdt,
            })
        if is_full_close and settings.enable_post_exit_followup and close_price > 0:
            follow_minutes = max(5, settings.post_exit_followup_minutes)
            item_id = f"followup:{position['symbol']}:{position.get('side', '')}:{int(time.time() * 1000)}"
            self.shadow_store.upsert_many([
                {
                    "id": item_id,
                    "symbol": position["symbol"],
                    "side": position.get("side", ""),
                    "sample_source": "post_exit_followup",
                    "start_ts": int(time.time() * 1000),
                    "expiry_ts": int(time.time() * 1000) + follow_minutes * 60 * 1000,
                    "entry_price": close_price,
                    "tp_price": close_price,
                    "sl_price": close_price,
                    "confidence": 0.0,
                    "setup_key": str(lifecycle_state.get("setup_key", position.get("setup_key", "")) or ""),
                    "market_regime": lifecycle_state.get("market_regime", position.get("market_regime", "unknown")),
                    "market_phase": lifecycle_state.get("market_phase", position.get("market_phase", "neutral_balance")),
                    "breakout_phase": lifecycle_state.get("breakout_phase", position.get("breakout_phase", "neutral")),
                    "trend_stage": lifecycle_state.get("trend_stage", position.get("trend_stage", "unknown")),
                    "learning_context_summary": lifecycle_state.get("learning_context_summary", position.get("learning_context_summary", "")),
                    "learning_adjustments": lifecycle_state.get("learning_adjustments", position.get("learning_adjustments", {})),
                    "entry_notional_usdt": round(entry_notional_usdt, 8),
                    "margin_used": round(margin_used, 8),
                    "leverage": leverage,
                    "effective_leverage_realized": round(effective_leverage_realized, 6),
                    "exit_reason": reason,
                    "horizon_bars": 0,
                    "created_at": close_time,
                }
            ])

    def close_position(self, position: Dict[str, Any], reason: str) -> Dict[str, Any]:
        side = "buy" if position["side"] == "short" else "sell"
        pos_side = self._pos_side(position.get("side", ""))
        size = max(float(position.get("size", 0.0) or 0.0), settings.lifecycle_min_position_size)
        result = self.client.safe_place_order(
            inst_id=position["symbol"],
            side=side,
            pos_side=pos_side,
            size=size,
            order_type="market",
            price=None,
            reduce_only=True,
            margin_mode=settings.td_mode,
            fallback_pos_side=position.get("side", ""),
        ) if settings.enable_live_execution else {"code": "0", "data": [{"ordId": f"paper-close-{position['symbol']}"}]}
        size = self._safe_float(result.get("normalized_size"), size)
        execution_snapshot = self._fetch_realized_close_snapshot(position, result, size)
        self.orders.append({
            "symbol": position["symbol"],
            "exit_reason": reason,
            "close_result": result,
            "realized_pnl": execution_snapshot.get("realized_pnl", 0.0),
            "close_price": execution_snapshot.get("close_price", 0.0),
            "fill_fee": execution_snapshot.get("fill_fee", 0.0),
        })
        self._append_trade_record(position, reason, size, "full_exit", "exit", execution_snapshot, is_full_close=True)
        self.protective.clear_symbol_pending_algos(position["symbol"], position.get("side", ""))
        self.lifecycle.clear(position["symbol"], position.get("side", ""))
        return {
            "symbol": position["symbol"],
            "reason": reason,
            "execution_mode": "live" if settings.enable_live_execution else "paper",
            "order_result": result,
            "realized_pnl": execution_snapshot.get("realized_pnl", 0.0),
            "close_price": execution_snapshot.get("close_price", 0.0),
            "realized_pnl_source": execution_snapshot.get("realized_pnl_source", "unknown"),
        }

    def partial_close_position(self, position: Dict[str, Any], reason: str, fraction: float) -> Dict[str, Any]:
        original_size = max(float(position.get("size", 0.0) or 0.0), settings.lifecycle_min_position_size)
        close_fraction = max(0.0, min(float(fraction or 0.0), 1.0))
        requested_size = max(original_size * close_fraction, settings.lifecycle_min_position_size)
        side = "buy" if position.get("side") == "short" else "sell"
        pos_side = self._pos_side(position.get("side", ""))

        result = self.client.safe_place_order(
            inst_id=position["symbol"],
            side=side,
            pos_side=pos_side,
            size=requested_size,
            order_type="market",
            price=None,
            reduce_only=True,
            margin_mode=settings.td_mode,
            fallback_pos_side=position.get("side", ""),
        ) if settings.enable_live_execution else {"code": "0", "data": [{"ordId": f"paper-partial-close-{position['symbol']}"}]}

        requested_size = self._safe_float(result.get("normalized_size"), requested_size)
        execution_snapshot = self._fetch_realized_close_snapshot(position, result, requested_size)
        filled_size = self._safe_float(execution_snapshot.get("filled_size"), requested_size)

        self.orders.append({
            "symbol": position["symbol"],
            "exit_reason": reason,
            "close_result": result,
            "realized_pnl": execution_snapshot.get("realized_pnl", 0.0),
            "close_price": execution_snapshot.get("close_price", 0.0),
            "fill_fee": execution_snapshot.get("fill_fee", 0.0),
            "partial": True,
            "requested_size": requested_size,
            "filled_size": filled_size,
        })

        management_action = str(position.get("management_action") or "partial_exit")
        lifecycle_stage = str(position.get("lifecycle_stage") or "partial_exit")
        self._append_trade_record(position, reason, filled_size, management_action, lifecycle_stage, execution_snapshot, is_full_close=False)

        remaining = max(original_size - filled_size, 0.0)
        state = self.lifecycle.get(position["symbol"], position.get("side", ""))
        if remaining <= settings.lifecycle_min_position_size:
            self.protective.clear_symbol_pending_algos(position["symbol"], position.get("side", ""))
            self.lifecycle.clear(position["symbol"], position.get("side", ""))
        else:
            state["partial_exit_count"] = int(state.get("partial_exit_count", 0) or 0) + 1
            if lifecycle_stage == "tp1":
                state["tp1_done"] = True
            if lifecycle_stage == "tp2":
                state["tp2_done"] = True
            state["last_partial_close_ts"] = time.time()
            state["remaining_size"] = remaining
            state["last_action"] = management_action
            state["last_reason"] = reason
            self.lifecycle.update(position["symbol"], position.get("side", ""), state)

        return {
            "symbol": position["symbol"],
            "reason": reason,
            "execution_mode": "live" if settings.enable_live_execution else "paper",
            "order_result": result,
            "partial": True,
            "requested_size": requested_size,
            "filled_size": filled_size,
            "remaining_size": remaining,
            "realized_pnl": execution_snapshot.get("realized_pnl", 0.0),
            "close_price": execution_snapshot.get("close_price", 0.0),
            "realized_pnl_source": execution_snapshot.get("realized_pnl_source", "unknown"),
        }
