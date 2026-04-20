import logging
import time
from typing import Any, Dict, List

from ai.adaptive_policy_store import AdaptivePolicyStore
from ai.autonomy_controller import AIAutonomyController
from ai.base_scorer import BaseScorer
from ai.ensemble_voter import EnsembleVoter
from ai.risk_guard_ai import RiskGuardAI
from analysis.breakout_analysis import BreakoutAnalyzer
from analysis.feature_builder import FeatureBuilder
from analysis.regime_detector import RegimeDetector
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.trend_analysis import TrendAnalyzer
from config.settings import settings
from services.account_service import AccountService
from services.ai_review_judge_service import AIReviewJudgeService
from services.autonomy_audit_service import AutonomyAuditService
from services.daily_trade_digest_service import DailyTradeDigestService
from services.dashboard_state_service import DashboardStateService
from services.gpt_advisor_service import GPTAdvisorService
from services.market_pipeline_service import MarketPipelineService
from services.optimization_apply_service import OptimizationApplyService
from services.order_execution_service import OrderExecutionService
from services.position_manager_service import PositionManagerService
from services.position_sync_service import PositionSyncService
from services.preflight_service import PreflightService
from services.protective_order_service import ProtectiveOrderService
from storage.trade_store import TradeStore
from services.dual_layer_ai_service import DualLayerAIService
from storage.live_position_snapshot_store import LivePositionSnapshotStore
from storage.position_lifecycle_store import PositionLifecycleStore
from storage.shadow_observation_store import ShadowObservationStore


class TradingRuntimeService:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use the project's current MarketPipelineService signature (no args)
        self.pipeline = MarketPipelineService()
        self.account = AccountService()
        self.position_sync = PositionSyncService()
        self.preflight = PreflightService()
        self.order_exec = OrderExecutionService()
        self.protect = ProtectiveOrderService()
        self.position_manager = PositionManagerService()
        self.dashboard = DashboardStateService()
        self.trade_store = TradeStore()
        self.dual_layer = DualLayerAIService()
        self.live_position_store = LivePositionSnapshotStore()
        self.lifecycle_store = PositionLifecycleStore()
        self.shadow_store = ShadowObservationStore()
        self.ai = AIAutonomyController()
        self.audit = AutonomyAuditService()
        self.base_scorer = BaseScorer()
        self.ensemble = EnsembleVoter()
        self.risk_guard = RiskGuardAI()
        self.technical = TechnicalAnalyzer()
        self.breakout = BreakoutAnalyzer()
        self.trend = TrendAnalyzer()
        self.regime = RegimeDetector()
        self.feature_builder = FeatureBuilder()
        self.policy_store = AdaptivePolicyStore()
        self.digest_service = DailyTradeDigestService()
        self.gpt = GPTAdvisorService()
        self.review_judge = AIReviewJudgeService()
        self.optimizer = OptimizationApplyService()

        self._scan_offset = 0
        self._cycle_count = 0

    def _timeframe_seconds(self) -> int:
        tf = str(settings.primary_timeframe or "15m").strip().lower()
        if not tf:
            return 900
        unit = tf[-1]
        try:
            value = int(tf[:-1])
        except Exception:
            return 900
        if unit == "m":
            return max(60, value * 60)
        if unit == "h":
            return max(3600, value * 3600)
        if unit == "d":
            return max(86400, value * 86400)
        return 900


    def _position_key(self, row: Dict[str, Any]) -> str:
        return f"{row.get('symbol', '')}:{row.get('side', 'net') or 'net'}"

    def _sync_live_closed_positions(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        previous = self.live_position_store.load()
        current = {self._position_key(row): {**row} for row in positions if isinstance(row, dict) and row.get('symbol')}
        synced: List[Dict[str, Any]] = []
        if settings.enable_live_close_sync:
            for key, prev in previous.items():
                if key in current:
                    continue
                last_seen_ts = float(prev.get('last_seen_ts', 0.0) or 0.0)
                if self._cycle_count > 1 and (not last_seen_ts or __import__('time').time() - last_seen_ts >= settings.live_close_sync_min_age_sec):
                    margin_used = float(prev.get('margin_used', 0.0) or 0.0)
                    pnl = float(prev.get('upl', 0.0) or 0.0)
                    pnl_on_margin_pct = (pnl / margin_used * 100.0) if margin_used > 0 else 0.0
                    trade_record = {
                        'symbol': prev.get('symbol'),
                        'side': prev.get('side'),
                        'pnl': pnl,
                        'pnl_net': pnl,
                        'pnl_amount': pnl,
                        'pnl_ratio': float(prev.get('upl_ratio', 0.0) or 0.0),
                        'pnl_on_margin_pct': round(pnl_on_margin_pct, 6),
                        'margin_used': margin_used,
                        'drawdown': float(prev.get('max_drawdown', 0.0) or 0.0),
                        'reason': 'live_position_disappeared_needs_reconcile',
                        'review_area': 'live_close_sync_audit',
                        'entry_confidence': float(prev.get('entry_confidence', 0.0) or 0.0),
                        'trend_bias': prev.get('trend_bias'),
                        'market_regime': prev.get('market_regime', 'unknown'),
                        'pre_breakout_score': float(prev.get('pre_breakout_score', 0.0) or 0.0),
                        'size': float(prev.get('size', 0.0) or 0.0),
                        'filled_size': float(prev.get('size', 0.0) or 0.0),
                        'size_multiplier': float(prev.get('size_multiplier', 1.0) or 1.0),
                        'leverage': float(prev.get('leverage', 0.0) or 0.0),
                        'margin_pct': float(prev.get('margin_pct', 0.0) or 0.0),
                        'entry_price': float(prev.get('entry_price', 0.0) or 0.0),
                        'close_price': float(prev.get('current_price', prev.get('entry_price', 0.0)) or 0.0),
                        'realized_pnl_source': 'position_sync_disappearance',
                        'management_action': 'sync_audit_only',
                        'protection_state': prev.get('protection_state', 'balanced'),
                        'lifecycle_stage': prev.get('lifecycle_stage', 'none'),
                        'learning_tier': 'sync_audit',
                        'count_in_learning': False,
                        'is_full_close': False,
                    }
                    self.trade_store.append(trade_record)
                    self.lifecycle_store.clear(str(prev.get('symbol', '')), str(prev.get('side', '')))
                    synced.append(trade_record)
        now = __import__('time').time()
        for row in current.values():
            row['last_seen_ts'] = now
        self.live_position_store.replace(current)
        return synced

    def _run_daily_review_loop(self) -> Dict[str, Any]:
        if not settings.enable_daily_gpt_review:
            return {"enabled": False, "reason": "daily_review_disabled"}

        current_policy = self.policy_store.load()
        status = self.gpt.status()
        day = self.gpt.target_review_day()

        if str(current_policy.get("last_review_day", "")) == day:
            return {
                "enabled": True,
                "reason": "already_reviewed_today",
                "day": day,
                "summary": current_policy.get("last_review_summary", ""),
                "gpt_available": self.gpt.available(),
                "next_run_local": status.get("next_run_local"),
            }

        if not self.gpt.in_daily_review_window():
            return {
                "enabled": True,
                "reason": "waiting_for_daily_window",
                "day": day,
                "gpt_available": self.gpt.available(),
                "next_run_local": status.get("next_run_local"),
            }

        digest = self.digest_service.build_digest(day)
        digest["drift_alert"] = current_policy.get("drift_alert", {})
        digest["lesson_memory"] = list(current_policy.get("lesson_memory", []) or [])
        digest["autonomy_profile"] = {
            "level": current_policy.get("autonomy_level", 0),
            "state": current_policy.get("autonomy_state", "seed"),
            "limits": current_policy.get("autonomy_limits", {}),
            "maturity_score": current_policy.get("autonomy_maturity_score", 0.0),
        }

        if int(digest.get("trade_count", 0) or 0) < settings.daily_review_min_trades:
            return {
                "enabled": True,
                "reason": "not_enough_trades",
                "day": day,
                "trade_count": digest.get("trade_count", 0),
                "gpt_available": self.gpt.available(),
                "next_run_local": status.get("next_run_local"),
            }

        if not self.gpt.available():
            return {"enabled": True, "reason": "gpt_not_ready", "day": day, "gpt_available": False, "next_run_local": status.get("next_run_local")}

        review = self.gpt.review_daily_trades(digest, current_policy)
        judge = self.review_judge.evaluate(digest, review)
        consensus: Dict[str, Any] = {
            "initial_review": review,
            "judge": judge,
            "consensus_summary": str(review.get("summary", "")),
            "final_recommendations": list(judge.get("accepted_recommendations", [])),
            "discussion_rounds": [],
            "gpt_available": True,
        }

        working_review = review
        working_judge = judge
        for round_index in range(1, settings.gpt_deliberation_rounds + 1):
            if not working_judge.get("needs_more_discussion"):
                break

            discussion = self.gpt.discuss_disagreement(
                digest,
                working_review,
                {
                    "judge_verdict": working_judge.get("verdict"),
                    "objections": working_judge.get("objections", []),
                },
                current_policy,
                round_index,
            )
            consensus["discussion_rounds"].append(discussion)
            working_review = {
                **working_review,
                "recommendations": discussion.get(
                    "updated_recommendations",
                    working_review.get("recommendations", []),
                ),
                "summary": discussion.get(
                    "consensus_summary",
                    working_review.get("summary", ""),
                ),
            }
            working_judge = self.review_judge.evaluate(digest, working_review)
            consensus["consensus_summary"] = str(
                working_review.get("summary", consensus["consensus_summary"])
            )
            consensus["final_recommendations"] = list(
                working_judge.get("accepted_recommendations", [])
            )

            if not discussion.get("needs_more_discussion", False) and not working_judge.get("needs_more_discussion"):
                break

        adjustments = self.gpt.recommend_adjustments(digest, consensus, current_policy)
        saved_policy = self.optimizer.apply(day, adjustments, consensus)
        self.trade_store.save_daily_review(day, {
            "day": day,
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "summary": saved_policy.get("last_review_summary", ""),
            "effective_trade_count": int(digest.get("trade_count", 0) or 0),
            "exploratory_trade_count": int(self.trade_store.summary(limit=50).get("exploratory_closed_count", 0) or 0),
            "digest": digest,
            "consensus": consensus,
            "adjustments": adjustments,
        })
        return {
            "enabled": True,
            "day": day,
            "digest": digest,
            "review": review,
            "judge": judge,
            "final_judge": working_judge,
            "consensus": consensus,
            "adjustments": adjustments,
            "saved_policy": saved_policy,
            "summary": saved_policy.get("last_review_summary", ""),
            "gpt_available": True,
            "next_run_local": status.get("next_run_local"),
        }

    def _bootstrap_softness(self, trade_summary: Dict[str, Any]) -> Dict[str, float]:
        total_closed = int(trade_summary.get("closed_count", trade_summary.get("total_count", 0)) or 0)
        if total_closed < 8:
            return {"entry_buffer": 0.10, "max_soft_entries": 2}
        if total_closed < 20:
            return {"entry_buffer": 0.06, "max_soft_entries": 1}
        return {"entry_buffer": 0.0, "max_soft_entries": 0}

    def _capital_position_cap(self, account_summary: Dict[str, Any]) -> int:
        equity = float(account_summary.get("equity", 0.0) or 0.0)
        available = float(account_summary.get("available_equity", account_summary.get("available", equity)) or 0.0)
        capital = max(min(equity, available if available > 0 else equity), 0.0)
        if capital <= 30:
            return 1
        if capital <= 80:
            return 2
        if capital <= 150:
            return 3
        if capital <= 300:
            return 4
        return max(1, settings.max_open_positions)

    def _policy_history(self, limit: int = 7) -> List[Dict[str, Any]]:
        rows = self.trade_store.latest_daily_reviews(limit=limit)
        history: List[Dict[str, Any]] = []
        for row in rows:
            adjustments = row.get("adjustments", {}) if isinstance(row.get("adjustments"), dict) else {}
            history.append({
                "day": str(row.get("day", "")),
                "summary": str(row.get("summary", "")),
                "effective_trade_count": int(row.get("effective_trade_count", 0) or 0),
                "exploratory_trade_count": int(row.get("exploratory_trade_count", 0) or 0),
                "entry_confidence_shift": adjustments.get("entry_confidence_shift"),
                "size_multiplier_bias": adjustments.get("size_multiplier_bias"),
                "leverage_bias": adjustments.get("leverage_bias"),
                "protection_profile": adjustments.get("protection_profile"),
                "position_management_profile": adjustments.get("position_management_profile"),
            })
        return history

    def _monthly_history(self, limit: int = 4) -> List[Dict[str, Any]]:
        rows = self.trade_store.latest_monthly_reviews(limit=limit)
        history: List[Dict[str, Any]] = []
        for row in rows:
            history.append(
                {
                    "review_key": str(row.get("review_key", "")),
                    "summary": str(row.get("summary", "")),
                    "effective_trade_count": int(row.get("effective_trade_count", 0) or 0),
                    "exploratory_trade_count": int(row.get("exploratory_trade_count", 0) or 0),
                }
            )
        return history

    def _derive_autonomy_profile(self, learning_memory: Dict[str, Any], account_summary: Dict[str, Any]) -> Dict[str, Any]:
        effective_count = int(learning_memory.get("effective_trade_count_total", 0) or 0)
        recent_count = int(learning_memory.get("recent_effective_trade_count", 0) or 0)
        recent_win_rate = float(learning_memory.get("recent_win_rate", 0.0) or 0.0)
        avg_margin_return = float(learning_memory.get("recent_avg_margin_return_pct", 0.0) or 0.0)
        loss_streak = int(learning_memory.get("recent_consecutive_losses", 0) or 0)
        followup_mix = {str(item.get("label", "")): int(item.get("count", 0) or 0) for item in learning_memory.get("post_exit_followup_mix", []) if isinstance(item, dict)}
        followup_total = sum(followup_mix.values())
        too_early_ratio = (followup_mix.get("too_early", 0) / max(followup_total, 1)) if followup_total else 0.0

        level = 0
        state = "seed"
        if effective_count >= 60 and recent_count >= 24 and recent_win_rate >= 0.58 and avg_margin_return >= 0.8 and loss_streak <= 2 and too_early_ratio <= 0.40:
            level = 3
            state = "adaptive_multi_position"
        elif effective_count >= 28 and recent_count >= 14 and recent_win_rate >= 0.54 and avg_margin_return >= 0.35 and loss_streak <= 3 and too_early_ratio <= 0.5:
            level = 2
            state = "validated_growth"
        elif effective_count >= 12 and recent_count >= 8 and recent_win_rate >= 0.5 and avg_margin_return >= 0.05 and loss_streak <= 4:
            level = 1
            state = "assisted_growth"

        equity = float(account_summary.get("equity", 0.0) or 0.0)
        if equity <= 10:
            max_positions = 1 if level <= 1 else 2
            max_entries = 1 if level <= 2 else 2
        elif equity <= 30:
            max_positions = min(3, 1 + level)
            max_entries = min(2, 1 + (1 if level >= 2 else 0))
        else:
            max_positions = min(settings.max_open_positions, max(1, 1 + level))
            max_entries = min(settings.max_live_entries_per_cycle, max(1, 1 + (1 if level >= 2 else 0)))

        return {
            "autonomy_level": level,
            "autonomy_state": state,
            "maturity_score": round(min(1.0, effective_count / 80.0 * 0.35 + recent_win_rate * 0.35 + max(0.0, min(avg_margin_return / 2.0, 1.0)) * 0.2 + max(0.0, 1.0 - too_early_ratio) * 0.1), 6),
            "too_early_ratio": round(too_early_ratio, 6),
            "autonomy_limits": {
                "max_open_positions": max(1, max_positions),
                "max_live_entries_per_cycle": max(1, max_entries),
                "max_simultaneous_entries": max(1, max_entries),
            },
        }

    def _derive_drift_alert(self, learning_memory: Dict[str, Any]) -> Dict[str, Any]:
        recent_count = int(learning_memory.get("recent_effective_trade_count", 0) or 0)
        recent_win_rate = float(learning_memory.get("recent_win_rate", 0.0) or 0.0)
        avg_margin_return = float(learning_memory.get("recent_avg_margin_return_pct", 0.0) or 0.0)
        loss_streak = int(learning_memory.get("recent_consecutive_losses", 0) or 0)
        too_early_ratio = float(learning_memory.get("too_early_ratio", 0.0) or 0.0)
        followup_sample_count = int(learning_memory.get("followup_sample_count", 0) or 0)
        exploration_failure_rate = float(learning_memory.get("exploration_failure_rate", 0.0) or 0.0)
        exploration_sample_count = int(learning_memory.get("recent_exploration_sample_count", 0) or 0)
        weak_setups = list(learning_memory.get("weak_setups", []) or [])
        worst_phases = list(learning_memory.get("worst_market_phases", []) or [])
        worst_notional = list(learning_memory.get("worst_notional_buckets", []) or [])
        worst_leverage = list(learning_memory.get("worst_leverage_buckets", []) or [])

        score = 0.0
        reasons: List[str] = []
        lessons: List[str] = []

        if recent_count >= 8 and recent_win_rate < 0.42:
            score += 0.28
            reasons.append("recent_effective_win_rate_weak")
            lessons.append(f"recent effective win rate is only {recent_win_rate:.0%}; tighten entries until edge recovers")
        if recent_count >= 8 and avg_margin_return < -0.35:
            score += 0.22
            reasons.append("recent_margin_return_negative")
            lessons.append(f"recent margin return is {avg_margin_return:.2f}%; reduce size while relearning")
        if loss_streak >= 4:
            score += 0.18
            reasons.append("loss_cluster")
            lessons.append(f"loss streak reached {loss_streak}; slow down and protect capital")
        if followup_sample_count >= 6 and too_early_ratio >= 0.55:
            score += 0.16
            reasons.append("exits_too_early")
            lessons.append(f"post-exit follow-up shows {too_early_ratio:.0%} too-early exits; let winners breathe more selectively")
        if exploration_sample_count >= 12 and exploration_failure_rate >= 0.68:
            score += 0.10
            reasons.append("exploration_edge_missing")
            lessons.append(f"recent exploration failure rate is {exploration_failure_rate:.0%}; narrow exploration focus")
        if weak_setups:
            score += 0.10
            weak_labels = ", ".join(str(item.get("setup_key", "")) for item in weak_setups[:2] if item.get("setup_key"))
            reasons.append("weak_setup_cluster")
            if weak_labels:
                lessons.append(f"weak setups detected: {weak_labels}; do not overweight them")
        if worst_phases:
            weakest_phase = worst_phases[0]
            if int(weakest_phase.get("count", 0) or 0) >= 3 and float(weakest_phase.get("avg_margin_return_pct", 0.0) or 0.0) < -0.25:
                score += 0.08
                reasons.append("weak_market_phase")
                lessons.append(f"market phase {weakest_phase.get('label', 'unknown')} is underperforming; reduce trust there")
        if worst_notional:
            weakest_notional = worst_notional[0]
            if int(weakest_notional.get("count", 0) or 0) >= 3 and float(weakest_notional.get("avg_margin_return_pct", 0.0) or 0.0) < -0.3:
                score += 0.06
                reasons.append("weak_notional_bucket")
                lessons.append(f"notional bucket {weakest_notional.get('label', 'unknown')} is weak; avoid sizing into it")
        if worst_leverage:
            weakest_leverage = worst_leverage[0]
            if int(weakest_leverage.get("count", 0) or 0) >= 3 and float(weakest_leverage.get("avg_margin_return_pct", 0.0) or 0.0) < -0.3:
                score += 0.06
                reasons.append("weak_leverage_bucket")
                lessons.append(f"leverage bucket {weakest_leverage.get('label', 'unknown')} is weak; reduce leverage bias")

        score = round(min(score, 1.0), 6)
        if score >= 0.72:
            severity = "critical"
        elif score >= 0.52:
            severity = "high"
        elif score >= 0.32:
            severity = "medium"
        elif score >= 0.18:
            severity = "low"
        else:
            severity = "none"

        if severity == "critical":
            guardrails = ["single_position_only", "tight_protection", "reduce_size_and_leverage"]
        elif severity == "high":
            guardrails = ["single_position_only", "defensive_management", "reduced_size"]
        elif severity == "medium":
            guardrails = ["guarded_entries", "reduced_aggression"]
        elif severity == "low":
            guardrails = ["watch_for_repeat_losses"]
        else:
            guardrails = []

        return {
            "severity": severity,
            "score": score,
            "reasons": reasons[:6],
            "guardrails": guardrails,
            "lessons": lessons[:6],
        }

    def _refresh_learning_state(self, account_summary: Dict[str, Any]) -> Dict[str, Any]:
        learning_memory = self.trade_store.learning_memory_snapshot()
        autonomy_profile = self._derive_autonomy_profile(learning_memory, account_summary)
        drift_alert = self._derive_drift_alert(learning_memory)
        limits = dict(autonomy_profile.get("autonomy_limits", {}) or {})
        drift_severity = str(drift_alert.get("severity", "none") or "none")
        if drift_severity in {"high", "critical"}:
            limits["max_open_positions"] = 1
            limits["max_live_entries_per_cycle"] = 1
            limits["max_simultaneous_entries"] = 1
            autonomy_profile["autonomy_state"] = "guardrail_recovery"
        elif drift_severity == "medium":
            limits["max_live_entries_per_cycle"] = 1
            limits["max_simultaneous_entries"] = 1
            limits["max_open_positions"] = max(1, min(int(limits.get("max_open_positions", 1) or 1), 1 if float(account_summary.get("equity", 0.0) or 0.0) <= 10 else 2))
            autonomy_profile["autonomy_state"] = f"{autonomy_profile.get('autonomy_state', 'seed')}_guarded"
        autonomy_profile["autonomy_limits"] = limits
        policy = self.policy_store.load()
        policy["learning_memory_summary"] = learning_memory
        policy["learning_memory_updated_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        policy["autonomy_level"] = autonomy_profile["autonomy_level"]
        policy["autonomy_state"] = autonomy_profile["autonomy_state"]
        policy["autonomy_limits"] = autonomy_profile["autonomy_limits"]
        policy["autonomy_maturity_score"] = autonomy_profile["maturity_score"]
        policy["autonomy_too_early_ratio"] = autonomy_profile["too_early_ratio"]
        policy["drift_alert"] = {
            "severity": drift_alert.get("severity", "none"),
            "score": drift_alert.get("score", 0.0),
            "reasons": drift_alert.get("reasons", []),
            "guardrails": drift_alert.get("guardrails", []),
        }
        policy["drift_score"] = drift_alert.get("score", 0.0)
        policy["lesson_memory"] = drift_alert.get("lessons", [])
        policy["last_lesson_update"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        policy["drift_guard_active"] = drift_severity in {"medium", "high", "critical"}
        saved = self.policy_store.save(policy)
        return saved

    def _tracked_symbols_for_learning(self, selected_symbols: List[Any], symbols: List[Any]) -> List[Any]:
        tracked = self.shadow_store.all()
        if not tracked:
            return selected_symbols
        selected_ids = {str(item.get("instId", item.get("symbol", item))) for item in selected_symbols}
        symbol_map = {}
        for item in symbols if isinstance(symbols, list) else []:
            symbol_id = str(item.get("instId", item.get("symbol", item))) if isinstance(item, dict) else str(item)
            symbol_map[symbol_id] = item
        extra: List[Any] = []
        for row in tracked:
            symbol = str(row.get("symbol", "") or "")
            if not symbol or symbol in selected_ids:
                continue
            extra.append(symbol_map.get(symbol, symbol))
            selected_ids.add(symbol)
            if len(extra) >= 6:
                break
        return selected_symbols + extra

    def _run_monthly_review_loop(self) -> Dict[str, Any]:
        if not settings.enable_monthly_gpt_review:
            return {"enabled": False, "reason": "monthly_review_disabled"}
        status = self.gpt.status()
        review_key = self.gpt.target_monthly_review_key()
        monthly_history = self.trade_store.latest_monthly_reviews(limit=1)
        if monthly_history and str(monthly_history[0].get("review_key", "")) == review_key:
            return {
                "enabled": True,
                "reason": "already_reviewed_this_month",
                "review_key": review_key,
                "summary": str(monthly_history[0].get("summary", "")),
            }
        if not self.gpt.in_monthly_review_window():
            return {"enabled": True, "reason": "waiting_for_monthly_window", "review_key": review_key, "next_run_local": status.get("next_run_local")}

        digest = self.digest_service.build_monthly_digest(review_key)
        if int(digest.get("trade_count", 0) or 0) < max(5, settings.daily_review_min_trades):
            return {"enabled": True, "reason": "not_enough_monthly_data", "review_key": review_key, "trade_count": digest.get("trade_count", 0)}

        review = self.gpt.review_monthly_progress(digest, self.policy_store.load())
        payload = {
            "review_key": review_key,
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "summary": str(review.get("summary", "")),
            "effective_trade_count": int(digest.get("effective_trade_count", 0) or 0),
            "exploratory_trade_count": int(digest.get("exploratory_trade_count", 0) or 0),
            "digest": digest,
            "review": review,
        }
        self.trade_store.save_monthly_review(review_key, payload)
        return {"enabled": True, "reason": "monthly_review_saved", "review_key": review_key, "summary": payload["summary"], "digest": digest, "review": review}

    def _resolve_shadow_observations(self, scans: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not settings.enable_shadow_learning:
            return {"enabled": False, "pending": 0, "resolved": 0}

        pending = self.shadow_store.all()
        if not pending:
            return {"enabled": True, "pending": 0, "resolved": 0}

        scan_map = {str(item.get("symbol", "")): item for item in scans if isinstance(item, dict)}
        survivors: List[Dict[str, Any]] = []
        resolved = 0

        for item in pending:
            symbol = str(item.get("symbol", "") or "")
            scan = scan_map.get(symbol)
            if not scan:
                survivors.append(item)
                continue
            df = scan.get("df")
            if df is None or df.empty or "ts" not in df:
                survivors.append(item)
                continue

            start_ts = int(item.get("start_ts", 0) or 0)
            expiry_ts = int(item.get("expiry_ts", 0) or 0)
            observed = df[df["ts"] >= start_ts]
            if observed.empty:
                survivors.append(item)
                continue

            last_ts = int(observed["ts"].iloc[-1])
            max_high = float(observed["high"].max())
            min_low = float(observed["low"].min())
            last_close = float(observed["close"].iloc[-1])
            entry_price = float(item.get("entry_price", 0.0) or 0.0)
            tp_price = float(item.get("tp_price", 0.0) or 0.0)
            sl_price = float(item.get("sl_price", 0.0) or 0.0)
            side = str(item.get("side", "long") or "long")
            sample_source = str(item.get("sample_source", "shadow_watch") or "shadow_watch")

            hit_tp = max_high >= tp_price if side == "long" else min_low <= tp_price
            hit_sl = min_low <= sl_price if side == "long" else max_high >= sl_price
            expired = expiry_ts > 0 and last_ts >= expiry_ts

            if not hit_tp and not hit_sl and not expired:
                survivors.append(item)
                continue

            if sample_source == "post_exit_followup":
                favorable_move_bp = (((max_high - entry_price) / max(entry_price, 1e-9)) * 10000.0) if side == "long" else (((entry_price - min_low) / max(entry_price, 1e-9)) * 10000.0)
                adverse_move_bp = (((entry_price - min_low) / max(entry_price, 1e-9)) * 10000.0) if side == "long" else (((max_high - entry_price) / max(entry_price, 1e-9)) * 10000.0)
                if favorable_move_bp >= settings.post_exit_early_threshold_bp and favorable_move_bp > adverse_move_bp:
                    result_label = "too_early"
                elif adverse_move_bp >= settings.post_exit_late_threshold_bp and adverse_move_bp > favorable_move_bp:
                    result_label = "well_timed"
                else:
                    result_label = "neutral"
                exit_price = last_close
                resolve_reason = "post_exit_followup_done"
                pnl_net = 0.0
                confidence_before = 0.0
                confidence_delta = 0.0
                sample = {
                    "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                    "symbol": symbol,
                    "setup_key": str(item.get("setup_key", "") or ""),
                    "side": side,
                    "market_regime": str(item.get("market_regime", "unknown") or "unknown"),
                    "market_phase": str(item.get("market_phase", "neutral_balance") or "neutral_balance"),
                    "breakout_phase": str(item.get("breakout_phase", "neutral") or "neutral"),
                    "trend_stage": str(item.get("trend_stage", "unknown") or "unknown"),
                    "learning_context_summary": str(item.get("learning_context_summary", "") or ""),
                    "confidence_before": confidence_before,
                    "confidence_delta": confidence_delta,
                    "confidence_after": confidence_before,
                    "result_label": result_label,
                    "leverage": float(item.get("leverage", 0.0) or 0.0),
                    "margin_used": float(item.get("margin_used", 0.0) or 0.0),
                    "entry_notional_usdt": float(item.get("entry_notional_usdt", 0.0) or 0.0),
                    "effective_leverage_realized": float(item.get("effective_leverage_realized", 0.0) or 0.0),
                    "pnl_on_margin_pct": 0.0,
                    "pnl_net": pnl_net,
                    "fee_usdt": 0.0,
                    "sample_source": sample_source,
                    "resolve_reason": resolve_reason,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "horizon_bars": int(item.get("horizon_bars", 0) or 0),
                    "followup_verdict": result_label,
                    "favorable_move_bp": round(favorable_move_bp, 4),
                    "adverse_move_bp": round(adverse_move_bp, 4),
                    "exit_reason": str(item.get("exit_reason", "") or ""),
                }
                self.trade_store.append_exploration_sample(sample)
                resolved += 1
                continue
            elif hit_tp and hit_sl:
                result_label = "loss"
                exit_price = sl_price
                resolve_reason = "both_hit_conservative_stop_first"
            elif hit_tp:
                result_label = "win"
                exit_price = tp_price
                resolve_reason = "tp_hit"
            elif hit_sl:
                result_label = "loss"
                exit_price = sl_price
                resolve_reason = "sl_hit"
            else:
                exit_price = last_close
                raw_return = ((exit_price - entry_price) / max(entry_price, 1e-9)) * (1.0 if side == "long" else -1.0)
                result_label = "win" if raw_return > 0 else "loss" if raw_return < 0 else "flat"
                resolve_reason = "time_expiry"

            pnl_net = ((exit_price - entry_price) / max(entry_price, 1e-9)) * (1.0 if side == "long" else -1.0)
            delta_base = settings.exploration_confidence_delta_win if pnl_net > 0 else settings.exploration_confidence_delta_loss if pnl_net < 0 else 0.0
            delta_scale = max(0.1, settings.shadow_learning_weight / max(settings.exploration_learning_weight, 1e-9))
            confidence_before = float(item.get("confidence", 0.0) or 0.0)
            confidence_delta = float(delta_base) * delta_scale

            sample = {
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "symbol": symbol,
                "setup_key": str(item.get("setup_key", "") or ""),
                "side": side,
                "market_regime": str(item.get("market_regime", "unknown") or "unknown"),
                "market_phase": str(item.get("market_phase", "neutral_balance") or "neutral_balance"),
                "breakout_phase": str(item.get("breakout_phase", "neutral") or "neutral"),
                "trend_stage": str(item.get("trend_stage", "unknown") or "unknown"),
                "learning_context_summary": str(item.get("learning_context_summary", "") or ""),
                "confidence_before": confidence_before,
                "confidence_delta": confidence_delta,
                "confidence_after": max(0.0, min(1.0, confidence_before + confidence_delta)),
                "result_label": result_label,
                "leverage": 0.0,
                "margin_used": 1.0,
                "entry_notional_usdt": 1.0,
                "effective_leverage_realized": 1.0,
                "pnl_on_margin_pct": round(pnl_net * 100.0, 6),
                "pnl_net": round(pnl_net, 6),
                "fee_usdt": 0.0,
                "sample_source": sample_source,
                "resolve_reason": resolve_reason,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "horizon_bars": int(item.get("horizon_bars", settings.shadow_observation_horizon_bars) or settings.shadow_observation_horizon_bars),
            }
            self.trade_store.append_exploration_sample(sample)
            self.dual_layer.record_setup_outcome(
                symbol=symbol,
                side=side,
                market_regime=str(item.get("market_regime", "unknown") or "unknown"),
                pnl_net=pnl_net,
                fee=0.0,
                margin=1.0,
                count_in_learning=False,
                effective_weight=settings.shadow_learning_weight,
            )
            resolved += 1

        self.shadow_store.replace(survivors)
        return {"enabled": True, "pending": len(survivors), "resolved": resolved}

    def _register_shadow_candidates(
        self,
        execution_candidates: List[Dict[str, Any]],
        selected_execution_candidates: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not settings.enable_shadow_learning:
            return {"enabled": False, "pending": 0, "registered": 0}

        existing = self.shadow_store.all()
        existing_ids = {str(item.get("id", "")) for item in existing}
        existing_keys = {f"{item.get('symbol', '')}:{item.get('side', '')}" for item in existing}
        live_keys = {f"{row.get('symbol', row.get('instId', ''))}:{row.get('side', row.get('posSide', ''))}" for row in positions if isinstance(row, dict)}
        selected_keys = {f"{item.get('symbol', '')}:{item.get('side', '')}" for item in selected_execution_candidates}

        timeframe_seconds = self._timeframe_seconds()
        registered: List[Dict[str, Any]] = []
        for candidate in execution_candidates:
            if len(registered) >= settings.max_shadow_candidates_per_cycle:
                break
            symbol = str(candidate.get("symbol", "") or "")
            side = str(candidate.get("side", "") or "")
            key = f"{symbol}:{side}"
            confidence = float(candidate.get("entry_decision", {}).get("confidence", 0.0) or 0.0)
            if confidence < settings.shadow_observation_min_confidence:
                continue
            if key in existing_keys or key in live_keys or key in selected_keys:
                continue
            if bool(candidate.get("preflight", {}).get("blocked", False)):
                continue

            entry_price = float(candidate.get("market_snapshot", {}).get("last_price", 0.0) or 0.0)
            atr = float(candidate.get("features", {}).get("atr", 0.0) or 0.0)
            if entry_price <= 0:
                continue
            stop_distance = max(entry_price * 0.003, atr * max(settings.shadow_stop_atr, 0.2))
            target_distance = stop_distance * max(settings.shadow_reward_risk, 1.0)
            if side == "long":
                tp_price = entry_price + target_distance
                sl_price = max(entry_price - stop_distance, entry_price * 0.92)
            else:
                tp_price = max(entry_price - target_distance, entry_price * 0.08)
                sl_price = entry_price + stop_distance

            start_ts = int(candidate.get("scan_ts", int(time.time() * 1000)) or int(time.time() * 1000))
            item_id = f"{symbol}:{side}:{start_ts}"
            if item_id in existing_ids:
                continue

            registered.append(
                {
                    "id": item_id,
                    "symbol": symbol,
                    "side": side,
                    "start_ts": start_ts,
                    "expiry_ts": start_ts + timeframe_seconds * 1000 * max(1, settings.shadow_observation_horizon_bars),
                    "entry_price": entry_price,
                    "tp_price": round(tp_price, 8),
                    "sl_price": round(sl_price, 8),
                    "confidence": confidence,
                    "setup_key": str(candidate.get("preflight", {}).get("setup_key", "") or ""),
                    "market_regime": str(candidate.get("features", {}).get("market_regime", "unknown") or "unknown"),
                    "market_phase": str(candidate.get("features", {}).get("market_phase", "neutral_balance") or "neutral_balance"),
                    "breakout_phase": str(candidate.get("features", {}).get("breakout_phase", "neutral") or "neutral"),
                    "trend_stage": str(candidate.get("features", {}).get("trend_stage", "unknown") or "unknown"),
                    "learning_context_summary": str(candidate.get("features", {}).get("learning_context_summary", "") or ""),
                    "horizon_bars": max(1, settings.shadow_observation_horizon_bars),
                    "sample_source": "shadow_watch",
                    "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                }
            )
            existing_keys.add(key)

        if registered:
            self.shadow_store.upsert_many(registered)
        return {"enabled": True, "pending": len(existing) + len(registered), "registered": len(registered)}

    def _select_execution_mix(self, candidates: List[Dict[str, Any]], allow_new_entries: int) -> List[Dict[str, Any]]:
        if allow_new_entries <= 0:
            return []
        effective = [c for c in candidates if str(c.get("preflight", {}).get("learning_tier", "")) == "effective"]
        exploration = [
            c for c in candidates
            if str(c.get("preflight", {}).get("learning_tier", "")) != "effective"
            and float(c.get("entry_decision", {}).get("confidence", 0.0) or 0.0) >= settings.exploration_min_confidence
        ]
        selected: List[Dict[str, Any]] = []
        reserved_effective = min(settings.min_effective_entries_per_cycle, len(effective), allow_new_entries)
        selected.extend(effective[:reserved_effective])

        remaining = allow_new_entries - len(selected)
        exploration_cap = min(settings.max_exploration_entries_per_cycle, remaining)
        if settings.enable_exploration_live_orders and exploration_cap > 0:
            selected.extend(exploration[:exploration_cap])

        remaining = allow_new_entries - len(selected)
        if remaining > 0:
            effective_tail = effective[reserved_effective:]
            selected.extend(effective_tail[:remaining])

        remaining = allow_new_entries - len(selected)
        if remaining > 0 and settings.enable_exploration_live_orders:
            used_symbols = {str(c.get("symbol", "")) for c in selected}
            exploration_tail = [c for c in exploration[exploration_cap:] if str(c.get("symbol", "")) not in used_symbols]
            selected.extend(exploration_tail[:remaining])
        return selected[:allow_new_entries]

    def _select_symbols_for_cycle(self, symbols: List[Any]) -> List[Any]:
        if not symbols:
            return []

        batch_size = int(getattr(settings, "scan_batch_size_per_cycle", 8) or 8)
        batch_size = max(3, min(batch_size, len(symbols)))

        start = self._scan_offset % len(symbols)
        end = start + batch_size
        if end <= len(symbols):
            selected = symbols[start:end]
        else:
            selected = symbols[start:] + symbols[: end - len(symbols)]

        self._scan_offset = (start + batch_size) % len(symbols)
        return selected

    def _build_watch_row(
        self,
        symbol: str,
        market_snapshot: Dict[str, Any],
        features: Dict[str, Any],
        entry_decision: Dict[str, Any],
        leverage_decision: Dict[str, Any],
        sizing_decision: Dict[str, Any],
        preflight: Dict[str, Any],
        decision_reason: str,
        block_reason: str,
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "last_price": market_snapshot.get("last_price"),
            "quote_volume_24h": market_snapshot.get("quote_volume"),
            "change_24h": market_snapshot.get("change_24h", 0.0),
            "trend_bias": features.get("trend_bias"),
            "trend_stage": features.get("trend_stage", "unknown"),
            "market_regime": features.get("market_regime"),
            "market_phase": features.get("market_phase", "neutral_balance"),
            "breakout_phase": features.get("breakout_phase", "neutral"),
            "entry_style_hint": features.get("entry_style_hint", "observe"),
            "learning_context_summary": features.get("learning_context_summary", ""),
            "market_story": features.get("market_story", ""),
            "entry_decision": {
                **entry_decision,
                "decision_reason": decision_reason,
                "block_reason": block_reason,
            },
            "leverage_decision": leverage_decision,
            "sizing_decision": sizing_decision,
            "preflight": preflight,
            "tp_sl": {
                "take_profit": features.get("suggested_take_profit", 0.0),
                "stop_loss": features.get("suggested_stop_loss", 0.0),
            },
            "market_basis_categories": entry_decision.get("market_basis_categories", []),
            "template_summary": entry_decision.get("template_summary", ""),
            "scan_debug": {
                "feature_strength": float(features.get("feature_strength", 0.0) or 0.0),
                "pre_breakout_score": float(features.get("pre_breakout_score", 0.0) or 0.0),
                "ensemble_confidence": float(features.get("ensemble_confidence", 0.0) or 0.0),
                "continuation_context_score": float(features.get("continuation_context_score", 0.0) or 0.0),
                "reversal_context_score": float(features.get("reversal_context_score", 0.0) or 0.0),
                "breakout_followthrough_score": float(features.get("breakout_followthrough_score", 0.0) or 0.0),
                "breakout_failure_risk": float(features.get("breakout_failure_risk", 0.0) or 0.0),
                "exhaustion_risk_score": float(features.get("exhaustion_risk_score", 0.0) or 0.0),
            },
        }

    def _run_preflight(
        self,
        candidate: Dict[str, Any],
        account_summary: Dict[str, Any],
        pos_mode: str,
    ) -> Dict[str, Any]:
        if hasattr(self.preflight, "check"):
            return self.preflight.check(candidate, account_summary, pos_mode)

        symbol = candidate["symbol"]
        market_snapshot = candidate.get("market_snapshot", {}) or {}
        leverage_decision = candidate.get("leverage_decision", {}) or {}
        sizing_decision = candidate.get("sizing_decision", {}) or {}

        leverage = int(leverage_decision.get("leverage", settings.default_leverage_min))
        margin_pct = float(leverage_decision.get("margin_pct", settings.default_margin_pct_min))
        size_multiplier = float(sizing_decision.get("size_multiplier", 1.0) or 1.0)
        last_price = float(market_snapshot.get("last_price", 0.0) or 0.0)
        available_usdt = float(
            account_summary.get("available_equity", account_summary.get("available", account_summary.get("equity", 0.0))) or 0.0
        )

        desired_margin = max(available_usdt * margin_pct * size_multiplier, 1.0)
        desired_notional = desired_margin * max(leverage, 1)
        desired_size = round(
            max(desired_notional / max(last_price, 1e-9), settings.lifecycle_min_position_size),
            8,
        )

        if hasattr(self.preflight, "preflight"):
            result = self.preflight.preflight(symbol, desired_size, last_price)
            if result.get("ok"):
                return {
                    "blocked": False,
                    "reason": result.get("max_avail_reason", "ok"),
                    "final_size": result.get("final_size"),
                    "final_price": result.get("final_price"),
                    "max_avail": result.get("max_avail"),
                    "max_avail_reason": result.get("max_avail_reason", "ok"),
                }
            return {
                "blocked": True,
                "reason": result.get("reason", "preflight_failed"),
                "max_avail_reason": result.get("max_avail_reason", "unknown"),
            }

        return {"blocked": False, "reason": "preflight_unavailable"}


    def _apply_gpt_live_overlay(self, candidate: Dict[str, Any], account_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not self.gpt.live_available():
            return candidate
        confidence = float(candidate.get("entry_decision", {}).get("confidence", 0.0) or 0.0)
        if confidence < settings.gpt_live_control_min_confidence:
            return candidate
        policy = self.policy_store.load()
        advice = self.gpt.advise_live_trade_candidate(candidate, policy, account_summary)
        candidate["gpt_live_advice"] = advice
        if not advice.get("enabled"):
            return candidate
        action = str(advice.get("action", "")).lower()
        if action in {"wait", "hold"}:
            candidate["entry_decision"]["action"] = "wait"
            candidate["entry_decision"]["decision_reason"] = f"gpt_live_{action}"
        elif action == "enter":
            candidate["entry_decision"]["action"] = "enter"
            candidate["entry_decision"]["decision_reason"] = "gpt_live_enter"
        boost = float(advice.get("confidence_boost", 0.0) or 0.0)
        candidate["entry_decision"]["confidence"] = round(max(0.0, min(1.0, float(candidate["entry_decision"].get("confidence", 0.0) or 0.0) + boost)), 6)
        if advice.get("leverage") not in (None, ""):
            candidate["leverage_decision"]["leverage"] = int(float(advice.get("leverage")))
        if advice.get("margin_pct") not in (None, ""):
            candidate["leverage_decision"]["margin_pct"] = float(advice.get("margin_pct"))
        candidate["features"]["gpt_market_structure"] = advice.get("market_structure", "unknown")
        candidate["features"]["gpt_tp_bias"] = float(advice.get("tp_bias", 0.0) or 0.0)
        candidate["features"]["gpt_sl_bias"] = float(advice.get("sl_bias", 0.0) or 0.0)
        return candidate

    def run_once(self) -> Dict[str, Any]:
        self._cycle_count += 1

        account_summary = self.account.summary()
        pos_mode = account_summary.get("pos_mode", "net")
        positions = self.position_sync.sync() if settings.enable_position_sync else []
        closed_sync_rows = self._sync_live_closed_positions(positions)
        trade_summary = self.trade_store.summary(limit=50)
        current_policy = self._refresh_learning_state(account_summary)
        risk_row = self.risk_guard.evaluate(account_summary)
        consecutive_losses = int(trade_summary.get("consecutive_losses", 0) or 0)

        if consecutive_losses >= settings.max_consecutive_losses_before_pause:
            risk_row = {**risk_row, "blocked": True, "reason": "max_consecutive_losses_reached"}

        autonomy = self.audit.run()
        symbols = self.pipeline.get_top_symbols()
        selected_symbols = self._select_symbols_for_cycle(symbols)
        selected_symbols = self._tracked_symbols_for_learning(selected_symbols, symbols)

        try:
            scans = self.pipeline.scan(selected_symbols)
        except Exception as exc:
            self.logger.exception("scan batch failed: symbols=%s error=%s", selected_symbols, exc)
            scans = []
        shadow_resolution = self._resolve_shadow_observations(scans)

        bootstrap = self._bootstrap_softness(trade_summary)
        feature_map: Dict[str, Dict[str, Any]] = {}
        watchlist: List[Dict[str, Any]] = []
        execution_candidates: List[Dict[str, Any]] = []
        soft_entry_used = 0
        total_closed = int(trade_summary.get("closed_count", trade_summary.get("total_count", 0)) or 0)

        for scan in scans:
            symbol = scan["symbol"]
            df = scan["df"]
            market_snapshot = scan["market_snapshot"]

            tech = self.technical.analyze(df)
            breakout = self.breakout.analyze(df)
            trend = self.trend.analyze(df)
            regime = self.regime.detect(df, tech, breakout, trend)
            base_out = self.base_scorer.score({**tech, **breakout, **trend, **regime})
            ensemble_out = self.ensemble.vote([base_out])

            features = self.feature_builder.build(
                symbol,
                market_snapshot,
                tech,
                breakout,
                trend,
                regime,
                ensemble_out,
            )
            feature_map[symbol] = features

            entry_decision = self.ai.decide_entry(features)
            leverage_decision = self.ai.decide_leverage(features)
            sizing_decision = self.ai.decide_sizing(features)

            setup_edge = self.dual_layer.get_setup_edge(
                symbol=symbol,
                side=str(entry_decision.get("side", "long")),
                market_regime=str(features.get("market_regime", "unknown") or "unknown"),
            )
            entry_decision["setup_memory"] = setup_edge
            entry_decision["confidence"] = round(
                max(0.0, min(1.0, float(entry_decision.get("confidence", 0.0) or 0.0) + float(setup_edge.get("confidence_bias", 0.0) or 0.0))),
                6,
            )
            entry_decision["effective_threshold"] = round(
                max(
                    settings.adaptive_min_trade_confidence_floor,
                    min(
                        settings.adaptive_min_trade_confidence_ceiling,
                        float(entry_decision.get("effective_threshold", settings.min_trade_confidence) or settings.min_trade_confidence)
                        + float(setup_edge.get("threshold_shift", 0.0) or 0.0),
                    ),
                ),
                4,
            )

            confidence = float(entry_decision.get("confidence", 0.0) or 0.0)
            base_threshold = float(
                entry_decision.get("effective_threshold", settings.min_trade_confidence)
                or settings.min_trade_confidence
            )

            if total_closed < 20:
                threshold = min(base_threshold, 0.35)
            elif total_closed < 50:
                threshold = min(base_threshold, 0.42)
            else:
                threshold = base_threshold

            soft_enter = (
                entry_decision.get("action") != "enter"
                and bootstrap["entry_buffer"] > 0
                and confidence >= max(settings.adaptive_min_trade_confidence_floor, threshold - bootstrap["entry_buffer"])
                and soft_entry_used < bootstrap["max_soft_entries"]
            )

            hard_enter = entry_decision.get("action") == "enter" or confidence >= threshold
            candidate_action = "enter" if hard_enter or soft_enter else "wait"

            decision_reason = "ai_enter"
            if entry_decision.get("action") != "enter" and hard_enter:
                decision_reason = "threshold_force_enter"
            if soft_enter:
                decision_reason = "bootstrap_soft_entry"
                soft_entry_used += 1

            candidate = {
                "symbol": symbol,
                "side": entry_decision.get("side", "long"),
                "scan_ts": int(df["ts"].iloc[-1]) if "ts" in df.columns and not df.empty else int(time.time() * 1000),
                "entry_decision": {
                    **entry_decision,
                    "action": candidate_action,
                    "original_action": entry_decision.get("action"),
                    "decision_reason": decision_reason,
                    "effective_threshold_runtime": threshold,
                    "setup_memory": setup_edge,
                },
                "leverage_decision": leverage_decision,
                "sizing_decision": sizing_decision,
                "features": features,
                "market_snapshot": market_snapshot,
            }

            if setup_edge.get("cooldown"):
                decision_reason = setup_edge.get("cooldown_reason", "setup_memory_cooldown")
                candidate["entry_decision"]["action"] = "wait"
                candidate["entry_decision"]["decision_reason"] = decision_reason
                candidate["entry_decision"]["block_reason"] = decision_reason
                candidate_action = "wait"

            if len(execution_candidates) < settings.gpt_live_control_top_n:
                candidate = self._apply_gpt_live_overlay(candidate, account_summary)

            preflight = self._run_preflight(candidate, account_summary, pos_mode)
            candidate["preflight"] = preflight

            block_reason = str(candidate["entry_decision"].get("block_reason", "") or "")
            if candidate_action != "enter" and not block_reason:
                block_reason = "ai_not_ready"
            elif preflight.get("blocked"):
                block_reason = preflight.get("reason", "preflight_blocked")

            watchlist.append(
                self._build_watch_row(
                    symbol=symbol,
                    market_snapshot=market_snapshot,
                    features=features,
                    entry_decision=candidate["entry_decision"],
                    leverage_decision=leverage_decision,
                    sizing_decision=sizing_decision,
                    preflight=preflight,
                    decision_reason=decision_reason,
                    block_reason=block_reason,
                )
            )

            self.logger.info(
                "[SCAN] cycle=%s symbol=%s action=%s raw_action=%s conf=%.4f thr=%.4f side=%s preflight=%s reason=%s",
                self._cycle_count,
                symbol,
                candidate_action,
                entry_decision.get("action"),
                confidence,
                threshold,
                candidate["side"],
                preflight.get("reason", "ok"),
                decision_reason,
            )

            if candidate_action == "enter" and not preflight.get("blocked"):
                execution_candidates.append(candidate)

        watchlist.sort(key=lambda x: float(x["entry_decision"].get("confidence", 0.0)), reverse=True)
        execution_candidates.sort(key=lambda x: float(x["entry_decision"].get("confidence", 0.0)), reverse=True)

        executed_orders: List[Dict[str, Any]] = []
        actual_orders: List[Dict[str, Any]] = []
        filtered_orders: List[Dict[str, Any]] = []
        protective_orders: List[Dict[str, Any]] = []
        capital_cap = self._capital_position_cap(account_summary)
        autonomy_limits = current_policy.get("autonomy_limits", {}) if isinstance(current_policy.get("autonomy_limits"), dict) else {}
        policy_max_open_positions = int(autonomy_limits.get("max_open_positions", settings.max_open_positions) or settings.max_open_positions)
        policy_max_live_entries = int(autonomy_limits.get("max_live_entries_per_cycle", settings.max_live_entries_per_cycle) or settings.max_live_entries_per_cycle)
        policy_max_simultaneous = int(autonomy_limits.get("max_simultaneous_entries", settings.max_simultaneous_entries) or settings.max_simultaneous_entries)
        open_slots = max(min(policy_max_open_positions, capital_cap) - len(positions), 0)
        allow_new_entries = min(open_slots, policy_max_live_entries, policy_max_simultaneous)
        selected_execution_candidates = self._select_execution_mix(execution_candidates, allow_new_entries)
        shadow_registration = self._register_shadow_candidates(execution_candidates, selected_execution_candidates, positions)

        if not risk_row.get("blocked") and allow_new_entries > 0:
            runtime_account_summary = dict(account_summary)
            for candidate in selected_execution_candidates:
                execution = self.order_exec.execute(candidate, pos_mode, runtime_account_summary)
                executed_orders.append(execution)
                if execution.get("order_success"):
                    actual_orders.append(execution)
                elif execution.get("execution_mode") == "filtered":
                    filtered_orders.append(execution)
                self.logger.info(
                    "[EXECUTE] cycle=%s symbol=%s mode=%s final_size=%s entry_price=%s preflight=%s",
                    self._cycle_count,
                    execution.get("symbol"),
                    execution.get("execution_mode"),
                    execution.get("final_size"),
                    execution.get("entry_price"),
                    candidate.get("preflight", {}).get("reason", "ok"),
                )
                if execution.get("order_success", execution.get("execution_mode") != "blocked"):
                    protective = self.protect.register(execution, pos_mode)
                    protective_orders.append(protective)
                    used_now = float(execution.get("effective_margin_used", 0.0) or 0.0)
                    runtime_account_summary["available_equity"] = max(0.0, float(runtime_account_summary.get("available_equity", runtime_account_summary.get("equity", 0.0)) or 0.0) - used_now)
                    self.logger.info(
                        "[PROTECT] cycle=%s symbol=%s tp=%s sl=%s",
                        self._cycle_count,
                        protective.get("symbol"),
                        protective.get("tp"),
                        protective.get("sl"),
                    )
                else:
                    order_msg = str(execution.get("order_result", {}))
                    if "51008" in order_msg or "insufficient" in order_msg.lower():
                        self.logger.warning("stop further entries this cycle after insufficient margin rejection: %s", execution.get("symbol"))
                        break
        elif risk_row.get("blocked"):
            self.logger.warning("new entries blocked by risk guard: %s", risk_row)

        managed_positions = self.position_manager.evaluate_positions(
            positions,
            feature_map,
            account_summary,
            pos_mode,
        )
        reflection = self.ai.reflect(self.trade_store.reflection_recent(limit=20))
        daily_review = self._run_daily_review_loop()
        monthly_review = self._run_monthly_review_loop()
        current_policy = self.policy_store.load()
        learning_database = {
            "storage": trade_summary.get("storage", {}),
            "setup_outcomes": self.trade_store.setup_outcomes_summary(limit=8),
            "policy_history": self._policy_history(limit=7),
            "monthly_history": self._monthly_history(limit=4),
            "learning_memory_summary": current_policy.get("learning_memory_summary", {}),
            "autonomy_profile": {
                "level": current_policy.get("autonomy_level", 0),
                "state": current_policy.get("autonomy_state", "seed"),
                "limits": current_policy.get("autonomy_limits", {}),
                "maturity_score": current_policy.get("autonomy_maturity_score", 0.0),
                "too_early_ratio": current_policy.get("autonomy_too_early_ratio", 0.0),
            },
            "drift_alert": current_policy.get("drift_alert", {}),
            "lesson_memory": current_policy.get("lesson_memory", []),
            "trade_scale_summary": {
                "avg_entry_notional_usdt": trade_summary.get("avg_entry_notional_usdt", 0.0),
                "avg_margin_used": trade_summary.get("avg_margin_used", 0.0),
                "avg_leverage": trade_summary.get("avg_leverage", 0.0),
                "avg_requested_leverage": trade_summary.get("avg_requested_leverage", 0.0),
                "avg_effective_leverage_realized": trade_summary.get("avg_effective_leverage_realized", 0.0),
                "avg_pnl_on_notional_bp": trade_summary.get("avg_pnl_on_notional_bp", 0.0),
                "avg_margin_return_pct": trade_summary.get("avg_margin_return_pct", 0.0),
            },
            "shadow_learning": {
                "enabled": settings.enable_shadow_learning,
                "resolved_this_cycle": shadow_resolution.get("resolved", 0),
                "registered_this_cycle": shadow_registration.get("registered", 0),
                "pending_count": shadow_registration.get("pending", shadow_resolution.get("pending", 0)),
            },
        }

        payload = {
            "autonomy_audit": autonomy,
            "balance": account_summary,
            "risk_guard": risk_row,
            "trade_summary": trade_summary,
            "storage": trade_summary.get("storage", {}),
            "pnl_today": {"unrealized": round(sum(float(x.get("upl", 0.0)) for x in positions), 4)},
            "watchlist": watchlist[:10],
            "positions": positions[: policy_max_open_positions],
            "executed_orders": executed_orders,
            "actual_orders": actual_orders,
            "filtered_orders": filtered_orders,
            "protective_orders": protective_orders,
            "managed_positions": managed_positions,
            "closed_sync_rows": closed_sync_rows,
            "ai_recent_learning_plain": reflection["plain_text"],
            "daily_gpt_review": daily_review,
            "monthly_gpt_review": monthly_review,
            "adaptive_policy": current_policy,
            "learning_database": learning_database,
            "policy_history": learning_database["policy_history"],
            "market_basis_ready": True,
            "gpt_connection": {
                **self.gpt.status(),
                "daily_review_enabled": settings.enable_daily_gpt_review,
                "daily_review_reason": daily_review.get("reason", "ok"),
                "live_control_enabled": self.gpt.live_available(),
            },
            "scan_meta": {
                "cycle": self._cycle_count,
                "selected_symbols": selected_symbols,
                "selected_count": len(selected_symbols),
                "scanned": len(selected_symbols),
                "watch_count": len(watchlist),
                "executed_count": len(executed_orders),
                "actual_order_count": len(actual_orders),
                "filtered_count": len(filtered_orders),
                "positions_count": len(positions),
                "total_top_symbols": len(symbols) if isinstance(symbols, list) else 0,
                "bootstrap_entry_buffer": bootstrap["entry_buffer"],
                "bootstrap_max_soft_entries": bootstrap["max_soft_entries"],
                "capital_position_cap": capital_cap,
                "policy_max_open_positions": policy_max_open_positions,
                "policy_max_live_entries": policy_max_live_entries,
                "allow_new_entries": allow_new_entries,
                "effective_candidates": len([c for c in execution_candidates if str(c.get("preflight", {}).get("learning_tier", "")) == "effective"]),
                "exploration_candidates": len([c for c in execution_candidates if str(c.get("preflight", {}).get("learning_tier", "")) != "effective"]),
                "selected_effective": len([c for c in selected_execution_candidates if str(c.get("preflight", {}).get("learning_tier", "")) == "effective"]),
                "selected_exploration": len([c for c in selected_execution_candidates if str(c.get("preflight", {}).get("learning_tier", "")) != "effective"]),
                "shadow_pending": shadow_registration.get("pending", shadow_resolution.get("pending", 0)),
                "shadow_registered": shadow_registration.get("registered", 0),
                "shadow_resolved": shadow_resolution.get("resolved", 0),
                "monthly_review_key": monthly_review.get("review_key", ""),
            },
            "system_notes": [
                "初期進場條件已放寬，但只作用在交易候選，不直接放鬆學習保護。",
                "小樣本保護、異常日保護、連敗保護、GPT 建議緩變保護仍保留。",
                "已改為分批輪掃 top symbols，避免一次全掃造成 429。",
                f"ENABLE_LIVE_EXECUTION={settings.enable_live_execution}",
                f"OKX_IS_DEMO={settings.okx_is_demo}",
                f"KILL_SWITCH={settings.kill_switch}",
                f"posMode={pos_mode}",
                f"tdMode={settings.td_mode}",
                f"GPT_AVAILABLE={self.gpt.available()}",
                f"GPT_MODEL={settings.gpt_model}",
                f"SQLITE_DB={trade_summary.get('storage', {}).get('db_path', '')}",
                f"BOOTSTRAP_ENTRY_BUFFER={bootstrap['entry_buffer']}",
                f"BOOTSTRAP_MAX_SOFT_ENTRIES={bootstrap['max_soft_entries']}",
                f"SHADOW_LEARNING={settings.enable_shadow_learning}",
                f"SHADOW_PENDING={shadow_registration.get('pending', shadow_resolution.get('pending', 0))}",
                f"SHADOW_REGISTERED={shadow_registration.get('registered', 0)}",
                f"SHADOW_RESOLVED={shadow_resolution.get('resolved', 0)}",
                f"MONTHLY_REVIEW={monthly_review.get('reason', '-')}",
                f"DRIFT_ALERT={current_policy.get('drift_alert', {}).get('severity', 'none')}",
                f"DRIFT_SCORE={current_policy.get('drift_score', 0.0)}",
                f"LESSONS={current_policy.get('lesson_memory', [])}",
                f"total_closed={total_closed}",
                f"closed_sync_rows={len(closed_sync_rows)}",
                f"consecutive_losses={trade_summary.get('consecutive_losses', 0)}/{settings.max_consecutive_losses_before_pause}",
                f"SCAN_BATCH={len(selected_symbols)}/{len(symbols) if isinstance(symbols, list) else 0}",
            ],
        }

        self.dashboard.update(payload)
        self.logger.info(
            "step40 done. cycle=%s scanned=%s/%s watch=%s positions=%s actual=%s filtered=%s executed=%s autonomy=%.2f risk_blocked=%s daily_review=%s gpt=%s",
            self._cycle_count,
            len(selected_symbols),
            len(symbols) if isinstance(symbols, list) else 0,
            len(watchlist),
            len(positions),
            len(actual_orders),
            len(filtered_orders),
            len(executed_orders),
            autonomy.get("autonomy_ratio", 0.0),
            risk_row.get("blocked"),
            daily_review.get("reason", daily_review.get("day", "ok")),
            self.gpt.available(),
        )
        return payload
