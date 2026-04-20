from typing import Any, Dict

from ai.adaptive_policy_store import AdaptivePolicyStore
from ai.tp_sl_advisor import TPSLAdvisor
from clients.okx_client import OKXClient
from config.settings import settings
from storage.order_store import OrderStore
from storage.position_lifecycle_store import PositionLifecycleStore
from services.dual_layer_ai_service import DualLayerAIService


class OrderExecutionService:
    def __init__(self) -> None:
        self.client = OKXClient()
        self.orders = OrderStore()
        self.policy_store = AdaptivePolicyStore()
        self.tp_sl = TPSLAdvisor()
        self.lifecycle = PositionLifecycleStore()
        self.dual_layer = DualLayerAIService()

    def _position_pos_side(self, pos_mode: str, side: str) -> str | None:
        return None if pos_mode == "net" and not settings.force_pos_side_in_net_mode else side

    def _min_required_margin(self, leverage: int) -> float:
        lev = max(int(leverage or 1), 1)
        return max(settings.min_margin_floor_usdt, settings.min_order_notional_usdt / lev)

    def _effective_margin_floor(self, equity: float) -> float:
        equity_floor = max(float(equity or 0.0), 0.0) * settings.effective_learning_min_margin_ratio
        leverage_anchor = max(int(settings.default_leverage_max or 1), 1)
        micro_cap = max(settings.min_margin_floor_usdt, (settings.min_order_notional_usdt / leverage_anchor) * 1.35)
        return max(settings.min_margin_floor_usdt, min(equity_floor, micro_cap))

    def _estimate_size(self, last_price: float, available_usdt: float, leverage: int, margin_pct: float, size_multiplier: float, equity: float) -> tuple[float, float, float]:
        usable_margin = max(0.0, available_usdt * max(0.05, min(settings.margin_reserve_ratio, 1.0)))
        desired_margin = usable_margin * margin_pct * size_multiplier
        desired_margin = max(desired_margin, self._min_required_margin(leverage))
        if not settings.enable_exploration_live_orders:
            desired_margin = max(desired_margin, self._effective_margin_floor(equity))
        desired_margin = min(desired_margin, usable_margin)
        desired_notional = max(desired_margin * max(leverage, 1), settings.min_order_notional_usdt)
        desired_size = round(max(desired_notional / max(last_price, 1e-9), settings.lifecycle_min_position_size), 8)
        return desired_size, desired_margin, desired_notional

    def _result_ok(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if str(result.get("code", "-1")) != "0":
            return False
        data = result.get("data") or []
        if not data:
            return True
        first = data[0] if isinstance(data[0], dict) else {}
        scode = str(first.get("sCode", "0") or "0")
        return scode in {"", "0"}

    def _learning_tier(self, equity: float, leverage: int, leverage_cap: int, desired_margin: float, expected_pnl_floor: float) -> str:
        # Live execution should not be blocked by a *predicted* pnl floor.
        # The realized abs pnl threshold belongs to post-close learning classification.
        effective_margin_floor = self._effective_margin_floor(equity)
        is_effective = (
            desired_margin >= effective_margin_floor
            and (not settings.effective_learning_force_max_leverage or leverage >= leverage_cap)
        )
        return "effective" if is_effective else "exploration"

    def _signal_learning_lane(self, candidate: Dict[str, Any], layer: Dict[str, Any]) -> str:
        entry = candidate.get("entry_decision", {}) or {}
        setup_memory = entry.get("setup_memory", {}) or {}
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        threshold = float(entry.get("effective_threshold_runtime", entry.get("effective_threshold", settings.min_trade_confidence)) or settings.min_trade_confidence)
        decision_reason = str(entry.get("decision_reason", "") or "")
        quality = str(setup_memory.get("quality", "unknown") or "unknown")
        sample_count = int(setup_memory.get("sample_count", 0) or 0)
        cooldown = bool(setup_memory.get("cooldown", False))
        confidence_buffer = max(0.0, settings.effective_learning_confidence_buffer)

        if cooldown:
            return "exploration"
        if decision_reason == "bootstrap_soft_entry":
            return "exploration"
        if quality == "weak" and sample_count >= settings.setup_memory_min_samples:
            return "exploration"
        if quality == "proven":
            return "effective"
        if confidence >= threshold + confidence_buffer and str(layer.get("learning_tier", "exploration")) == "effective":
            return "effective"
        return "exploration"

    def execute(self, candidate: Dict[str, Any], pos_mode: str, account_summary: Dict[str, Any]) -> Dict[str, Any]:
        preflight = candidate.get("preflight", {})
        if preflight.get("blocked"):
            return {"symbol": candidate["symbol"], "side": candidate["side"], "execution_mode": "blocked", "reason": preflight.get("reason", "preflight_blocked"), "preflight": preflight, "order_success": False}

        leverage_decision = candidate.get("leverage_decision", {})
        sizing_decision = candidate.get("sizing_decision", {})
        policy = self.policy_store.load()
        requested_leverage = int(leverage_decision.get("leverage", settings.default_leverage_min) or settings.default_leverage_min)
        margin_pct = float(leverage_decision.get("margin_pct", settings.default_margin_pct_min) or settings.default_margin_pct_min)
        size_multiplier = float(sizing_decision.get("size_multiplier", 1.0) or 1.0)
        last_price = float(candidate.get("market_snapshot", {}).get("last_price", 0.0) or 0.0)
        equity = float(account_summary.get("equity", 0.0) or 0.0)
        available_usdt = float(account_summary.get("available_equity", account_summary.get("equity", 0.0)) or 0.0)
        leverage_cap = max(int(self.client.safe_get_max_leverage(candidate["symbol"], settings.td_mode, self._position_pos_side(pos_mode, candidate["side"]))), 1) if settings.enable_live_execution else int(settings.default_leverage_max)
        leverage = max(1, min(requested_leverage, leverage_cap))
        if settings.effective_learning_force_max_leverage:
            leverage = leverage_cap
        desired_size, desired_margin, desired_notional = self._estimate_size(last_price, available_usdt, leverage, margin_pct, size_multiplier, equity)
        expected_pnl_floor = desired_notional * settings.hard_stop_loss_pct
        layer = self.dual_layer.classify_entry(symbol=candidate["symbol"], side=candidate["side"], market_regime=str(candidate.get("features", {}).get("market_regime", "unknown")), confidence=float(candidate.get("entry_decision", {}).get("confidence", 0.0) or 0.0), leverage=leverage, leverage_cap=leverage_cap, desired_margin=desired_margin, equity=equity)
        learning_tier = self._signal_learning_lane(candidate, layer)
        learning_enabled = learning_tier == "effective"
        if not learning_enabled and settings.enable_exploration_live_orders:
            exploration_scale = max(0.2, min(settings.exploration_order_size_scale, 1.0))
            desired_margin = max(self._min_required_margin(leverage), desired_margin * exploration_scale)
            desired_notional = max(desired_margin * max(leverage, 1), settings.min_order_notional_usdt)
            desired_size = round(max(desired_notional / max(last_price, 1e-9), settings.lifecycle_min_position_size), 8)
        if not learning_enabled and not settings.enable_exploration_live_orders:
            filter_detail = {
                "effective_margin_floor": round(layer["effective_margin_floor"], 6),
                "desired_margin": round(desired_margin, 6),
                "requested_leverage": requested_leverage,
                "applied_leverage": leverage,
                "leverage_cap": leverage_cap,
                "force_max_leverage": settings.effective_learning_force_max_leverage,
            }
            execution = {
                "symbol": candidate["symbol"],
                "side": candidate["side"],
                "execution_mode": "filtered",
                "order_success": False,
                "reason": "exploration_filtered_margin_or_leverage",
                "learning_tier": learning_tier,
                "setup_key": layer["setup_key"],
                "expected_pnl_floor": expected_pnl_floor,
                "desired_margin": desired_margin,
                "desired_notional": desired_notional,
                "requested_leverage": requested_leverage,
                "leverage_cap": leverage_cap,
                "leverage": leverage,
                "margin_pct": margin_pct,
                "preflight": preflight,
                "entry_price": last_price,
                "final_size": float(preflight.get("final_size") or desired_size),
                "effective_margin_used": desired_margin,
                "pos_side_used": self._position_pos_side(pos_mode, candidate["side"]) or "net/none",
                "filter_detail": filter_detail,
            }
            self.orders.append(execution)
            return execution

        preflight_final_size = preflight.get("final_size")
        if preflight_final_size is None:
            preflight_final_size = preflight.get("adjusted_size")
        final_size = float(preflight_final_size if preflight_final_size not in (None, "", 0, 0.0) else desired_size)
        if not learning_enabled and settings.enable_exploration_live_orders:
            final_size = max(settings.lifecycle_min_position_size, final_size * max(0.2, min(settings.exploration_order_size_scale, 1.0)))

        preflight_entry_price = preflight.get("final_price")
        entry_price = float(preflight_entry_price if preflight_entry_price not in (None, "") else last_price)

        pos_side = self._position_pos_side(pos_mode, candidate["side"])
        order_side = "buy" if candidate["side"] == "long" else "sell"

        leverage_result = {"code": "0", "data": []}
        if settings.enable_live_execution and settings.set_leverage_before_entry:
            leverage_result = self.client.safe_set_leverage(inst_id=candidate["symbol"], leverage=leverage, margin_mode=settings.td_mode, pos_side=pos_side, fallback_pos_side=candidate["side"])
            leverage = int(leverage_result.get("applied_leverage", leverage) or leverage)

        result = self.client.safe_place_order(
            inst_id=candidate["symbol"],
            side=order_side,
            pos_side=pos_side,
            size=final_size,
            order_type="market",
            price=None,
            reduce_only=False,
            margin_mode=settings.td_mode,
            fallback_pos_side=candidate["side"],
        ) if settings.enable_live_execution else {"code": "0", "data": [{"ordId": f"paper-{candidate['symbol']}"}]}
        order_success = self._result_ok(result)
        final_size = float(result.get("normalized_size", final_size) or final_size)

        tp_sl = self.tp_sl.suggest(candidate.get("features", {}), candidate["side"], float(candidate.get("entry_decision", {}).get("confidence", 0.0) or 0.0))
        entry_decision = candidate.get("entry_decision", {}) or {}
        sizing_decision = candidate.get("sizing_decision", {}) or {}
        leverage_decision = candidate.get("leverage_decision", {}) or {}
        learning_adjustments = {
            "entry": entry_decision.get("learning_adjustment", {}),
            "sizing": sizing_decision.get("learning_adjustment", {}),
            "leverage": leverage_decision.get("learning_adjustment", {}),
        }
        execution = {
            "symbol": candidate["symbol"],
            "side": candidate["side"],
            "execution_mode": ("live" if settings.enable_live_execution else "paper") if order_success else "failed",
            "order_result": result,
            "order_success": order_success,
            "leverage_result": leverage_result,
            "leverage_success": self._result_ok(leverage_result),
            "desired_size": desired_size,
            "final_size": final_size,
            "entry_price": entry_price,
            "size_multiplier": size_multiplier,
            "requested_leverage": requested_leverage,
            "leverage_cap": leverage_cap,
            "leverage": leverage,
            "margin_pct": margin_pct,
            "desired_margin": desired_margin,
            "desired_notional": desired_notional,
            "effective_margin_used": max(desired_margin, settings.min_margin_floor_usdt),
            "entry_confidence": float(entry_decision.get("confidence", 0.0) or 0.0),
            "trend_bias": candidate.get("features", {}).get("trend_bias"),
            "trend_stage": candidate.get("features", {}).get("trend_stage", "unknown"),
            "market_regime": candidate.get("features", {}).get("market_regime", "unknown"),
            "market_phase": candidate.get("features", {}).get("market_phase", "neutral_balance"),
            "breakout_phase": candidate.get("features", {}).get("breakout_phase", "neutral"),
            "pre_breakout_score": float(candidate.get("features", {}).get("pre_breakout_score", 0.0) or 0.0),
            "breakout_followthrough_score": float(candidate.get("features", {}).get("breakout_followthrough_score", 0.0) or 0.0),
            "breakout_failure_risk": float(candidate.get("features", {}).get("breakout_failure_risk", 0.0) or 0.0),
            "rebound_probability_score": float(candidate.get("features", {}).get("rebound_probability_score", 0.0) or 0.0),
            "exhaustion_risk_score": float(candidate.get("features", {}).get("exhaustion_risk_score", 0.0) or 0.0),
            "continuation_context_score": float(candidate.get("features", {}).get("continuation_context_score", 0.0) or 0.0),
            "reversal_context_score": float(candidate.get("features", {}).get("reversal_context_score", 0.0) or 0.0),
            "entry_style_hint": candidate.get("features", {}).get("entry_style_hint", "observe"),
            "learning_context_summary": candidate.get("features", {}).get("learning_context_summary", ""),
            "market_story": candidate.get("features", {}).get("market_story", ""),
            "market_basis_summary": entry_decision.get("template_summary", ""),
            "market_basis_snapshot": entry_decision.get("market_basis_snapshot", {}),
            "learning_adjustments": learning_adjustments,
            "review_area": "entry",
            "exit_style": policy.get("exit_style", "balanced"),
            "protection_profile": policy.get("protection_profile", "balanced"),
            "position_management_profile": policy.get("position_management_profile", "balanced"),
            "management_action": "entry",
            "protection_state": policy.get("protection_profile", "balanced"),
            "tp_sl": tp_sl,
            "preflight": preflight,
            "pos_side_used": pos_side or "net/none",
            "learning_tier": learning_tier,
            "count_in_learning": learning_tier == "effective",
            "learning_weight": settings.effective_learning_weight if learning_tier == "effective" else settings.exploration_learning_weight,
            "effective_margin_floor": layer["effective_margin_floor"],
            "setup_key": layer["setup_key"],
            "expected_pnl_floor": round(expected_pnl_floor, 6),
            "entry_time": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        }
        if order_success:
            self.lifecycle.update(candidate["symbol"], candidate["side"], {
                "scale_in_count": 0,
                "partial_exit_count": 0,
                "tp1_done": False,
                "tp2_done": False,
                "last_action": "entry",
                "last_reason": "new_position",
                "entry_time": execution["entry_time"],
                "entry_price": entry_price,
                "requested_leverage": requested_leverage,
                "leverage_cap": leverage_cap,
                "applied_leverage": leverage,
                "margin_used": execution["effective_margin_used"],
                "desired_margin": desired_margin,
                "desired_notional": desired_notional,
                "learning_tier": learning_tier,
                "setup_key": layer["setup_key"],
                "count_in_learning": learning_tier == "effective",
                "size_multiplier": size_multiplier,
                "entry_confidence": execution["entry_confidence"],
                "trend_bias": execution["trend_bias"],
                "trend_stage": execution["trend_stage"],
                "market_regime": execution["market_regime"],
                "market_phase": execution["market_phase"],
                "breakout_phase": execution["breakout_phase"],
                "pre_breakout_score": execution["pre_breakout_score"],
                "breakout_followthrough_score": execution["breakout_followthrough_score"],
                "breakout_failure_risk": execution["breakout_failure_risk"],
                "rebound_probability_score": execution["rebound_probability_score"],
                "exhaustion_risk_score": execution["exhaustion_risk_score"],
                "continuation_context_score": execution["continuation_context_score"],
                "reversal_context_score": execution["reversal_context_score"],
                "entry_style_hint": execution["entry_style_hint"],
                "learning_context_summary": execution["learning_context_summary"],
                "market_story": execution["market_story"],
                "market_basis_summary": execution["market_basis_summary"],
                "market_basis_snapshot": execution["market_basis_snapshot"],
                "learning_adjustments": learning_adjustments,
                "pos_side_used": execution["pos_side_used"],
            })
        self.orders.append(execution)
        return execution

    def manage_position(self, symbol: str, side: str, pos_mode: str, current_size: float, fraction: float, action: str, features: Dict[str, Any]) -> Dict[str, Any]:
        fraction = max(0.0, float(fraction or 0.0))
        target_size = round(max(settings.lifecycle_min_position_size, current_size * fraction), 8)
        if action == "scale_in":
            order_side = "buy" if side == "long" else "sell"
            reduce_only = False
        else:
            order_side = "sell" if side == "long" else "buy"
            reduce_only = True
        pos_side = self._position_pos_side(pos_mode, side)
        result = self.client.safe_place_order(inst_id=symbol, side=order_side, pos_side=pos_side, size=target_size, order_type="market", price=None, reduce_only=reduce_only, margin_mode=settings.td_mode, fallback_pos_side=side) if settings.enable_live_execution else {"code": "0", "data": [{"ordId": f"paper-manage-{symbol}-{action}"}]}
        target_size = float(result.get("normalized_size", target_size) or target_size)
        state = self.lifecycle.get(symbol, side)
        updates = {"last_action": action, "last_reason": action}
        if action == "scale_in":
            updates["scale_in_count"] = int(state.get("scale_in_count", 0) or 0) + 1
        record = {
            "symbol": symbol,
            "side": side,
            "execution_mode": "live" if settings.enable_live_execution else "paper",
            "management_action": action,
            "managed_size": target_size,
            "fraction": fraction,
            "order_result": result,
            "trend_bias": features.get("trend_bias"),
            "trend_stage": features.get("trend_stage", "unknown"),
            "market_regime": features.get("market_regime", "unknown"),
            "market_phase": features.get("market_phase", "neutral_balance"),
            "breakout_phase": features.get("breakout_phase", "neutral"),
            "pre_breakout_score": float(features.get("pre_breakout_score", 0.0) or 0.0),
            "breakout_followthrough_score": float(features.get("breakout_followthrough_score", 0.0) or 0.0),
            "breakout_failure_risk": float(features.get("breakout_failure_risk", 0.0) or 0.0),
            "exhaustion_risk_score": float(features.get("exhaustion_risk_score", 0.0) or 0.0),
            "review_area": "position_management",
        }
        self.lifecycle.update(symbol, side, updates)
        self.orders.append(record)
        return record
