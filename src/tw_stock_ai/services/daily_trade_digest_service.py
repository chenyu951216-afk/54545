from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict
from zoneinfo import ZoneInfo

from config.settings import settings
from storage.trade_store import TradeStore


class DailyTradeDigestService:
    """
    Safe compact digest:
    - no huge per-trade payload to GPT
    - only fixed-size summary fields
    """

    def __init__(self) -> None:
        self.store = TradeStore()

    def today_key(self) -> str:
        return datetime.now(ZoneInfo(settings.gpt_review_timezone)).date().isoformat()

    def _rows_for_day(self, rows: list[Dict[str, Any]], day: str) -> list[Dict[str, Any]]:
        tz = ZoneInfo(settings.gpt_review_timezone)
        out: list[Dict[str, Any]] = []
        for row in rows:
            ts = str(row.get("timestamp", row.get("close_time", "")) or "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.tzinfo is None:
                continue
            if dt.astimezone(tz).date().isoformat() == day:
                out.append(row)
        return out

    def _top_counts(self, rows: list[Dict[str, Any]], field: str, limit: int = 6) -> list[Dict[str, Any]]:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            label = str(row.get(field, "") or "").strip()
            if not label:
                continue
            counts[label] += 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [{"label": label, "count": count} for label, count in ordered[:limit]]

    def _performance_by_field(self, rows: list[Dict[str, Any]], field: str, limit: int = 6) -> list[Dict[str, Any]]:
        grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "pnl_sum": 0.0})
        for row in rows:
            label = str(row.get(field, "") or "").strip()
            if not label:
                continue
            pnl = float(row.get("pnl_net", row.get("pnl", 0.0)) or 0.0)
            grouped[label]["count"] += 1.0
            grouped[label]["wins"] += 1.0 if pnl > 0 else 0.0
            grouped[label]["pnl_sum"] += pnl
        ordered = sorted(grouped.items(), key=lambda item: (-item[1]["count"], -item[1]["pnl_sum"], item[0]))
        out: list[Dict[str, Any]] = []
        for label, stats in ordered[:limit]:
            count = max(stats["count"], 1.0)
            out.append(
                {
                    "label": label,
                    "count": int(stats["count"]),
                    "win_rate": round(stats["wins"] / count, 6),
                    "avg_pnl": round(stats["pnl_sum"] / count, 6),
                }
            )
        return out

    def _mean_value(self, rows: list[Dict[str, Any]], field: str, default: float = 0.0) -> float:
        values = [float(row.get(field, default) or default) for row in rows if row.get(field) not in (None, "")]
        return round(mean(values), 6) if values else default

    def _bucket_value(self, value: float, kind: str) -> str:
        if kind == "leverage":
            if value < 10:
                return "<10x"
            if value < 25:
                return "10-25x"
            if value < 50:
                return "25-50x"
            return "50x+"
        if kind == "margin":
            if value < 0.2:
                return "<0.2u"
            if value < 0.5:
                return "0.2-0.5u"
            if value < 1.0:
                return "0.5-1u"
            return "1u+"
        if kind == "notional":
            if value < 10:
                return "<10u"
            if value < 20:
                return "10-20u"
            if value < 50:
                return "20-50u"
            return "50u+"
        return "other"

    def _performance_by_bucket(self, rows: list[Dict[str, Any]], field: str, kind: str) -> list[Dict[str, Any]]:
        grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "pnl_margin_sum": 0.0, "pnl_sum": 0.0})
        for row in rows:
            raw = row.get(field)
            if raw in (None, ""):
                continue
            label = self._bucket_value(float(raw or 0.0), kind)
            pnl = float(row.get("pnl_net", row.get("pnl", 0.0)) or 0.0)
            pnl_margin = float(row.get("pnl_on_margin_pct", 0.0) or 0.0)
            grouped[label]["count"] += 1.0
            grouped[label]["wins"] += 1.0 if pnl > 0 else 0.0
            grouped[label]["pnl_margin_sum"] += pnl_margin
            grouped[label]["pnl_sum"] += pnl
        ordered = sorted(grouped.items(), key=lambda item: (-item[1]["count"], item[0]))
        return [
            {
                "label": label,
                "count": int(stats["count"]),
                "win_rate": round(stats["wins"] / max(stats["count"], 1.0), 6),
                "avg_pnl": round(stats["pnl_sum"] / max(stats["count"], 1.0), 6),
                "avg_pnl_on_margin_pct": round(stats["pnl_margin_sum"] / max(stats["count"], 1.0), 6),
            }
            for label, stats in ordered
        ]

    def build_digest(self, day: str | None = None) -> Dict[str, Any]:
        target_day = day or self.today_key()
        effective_rows = self.store.records_for_day(target_day, settings.gpt_review_timezone, effective_only=True)
        all_rows = self.store.records_for_day(target_day, settings.gpt_review_timezone, effective_only=False)
        exploratory_rows = [r for r in all_rows if not bool(r.get("count_in_learning", False)) and bool(r.get("is_full_close", True))]
        exploration_samples = self._rows_for_day(self.store.recent_exploration(limit=5000), target_day)
        setup_summary = self.store.setup_outcomes_summary(limit=8)

        if not all_rows and not exploration_samples:
            return {
                "day": target_day,
                "trade_count": 0,
                "effective_trade_count": 0,
                "exploratory_trade_count": 0,
                "exploration_sample_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "avg_drawdown": 0.0,
                "avg_pnl_on_margin_pct": 0.0,
                "avg_pnl_on_notional_bp": 0.0,
                "avg_margin_used": 0.0,
                "avg_entry_notional_usdt": 0.0,
                "avg_leverage": 0.0,
                "avg_requested_leverage": 0.0,
                "avg_effective_leverage_realized": 0.0,
                "avg_hold_minutes": 0.0,
                "effective_learning_weight": 0.0,
                "exploration_win_rate": 0.0,
                "top_setups": [],
                "market_regime_mix": [],
                "market_phase_mix": [],
                "breakout_phase_mix": [],
                "trend_stage_mix": [],
                "market_phase_performance": [],
                "exploration_source_mix": [],
                "leverage_bucket_performance": [],
                "margin_bucket_performance": [],
                "notional_bucket_performance": [],
                "exit_followup_mix": [],
            }

        pnls = [float(r.get("pnl_net", r.get("pnl", 0.0)) or 0.0) for r in effective_rows]
        fees = [abs(float(r.get("fee_usdt", r.get("fill_fee", 0.0)) or 0.0)) for r in effective_rows]
        dds = [float(r.get("drawdown", 0.0) or 0.0) for r in effective_rows]
        wins = len([p for p in pnls if p > 0])
        losses = len(pnls) - wins
        exploration_wins = len([r for r in exploration_samples if str(r.get("result_label", "")).lower() in {"win", "positive", "success", "effective"}])
        followups = [r for r in exploration_samples if str(r.get("sample_source", "") or "") == "post_exit_followup"]
        effective_learning_weight = sum(float(r.get("learning_weight", 1.0 if bool(r.get("count_in_learning")) else settings.exploration_learning_weight) or 0.0) for r in all_rows)

        return {
            "day": target_day,
            "trade_count": len(effective_rows),
            "effective_trade_count": len(effective_rows),
            "exploratory_trade_count": len(exploratory_rows),
            "exploration_sample_count": len(exploration_samples),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(len(effective_rows), 1), 6) if effective_rows else 0.0,
            "avg_pnl": round(mean(pnls), 6) if effective_rows else 0.0,
            "total_pnl": round(sum(pnls), 6),
            "total_fees": round(sum(fees), 6),
            "avg_drawdown": round(mean(dds), 6) if effective_rows else 0.0,
            "avg_pnl_on_margin_pct": self._mean_value(effective_rows, "pnl_on_margin_pct", 0.0),
            "avg_pnl_on_notional_bp": self._mean_value(effective_rows, "pnl_on_notional_bp", 0.0),
            "avg_margin_used": self._mean_value(all_rows, "margin_used", 0.0),
            "avg_entry_notional_usdt": self._mean_value(all_rows, "entry_notional_usdt", 0.0),
            "avg_leverage": self._mean_value(all_rows, "leverage", 0.0),
            "avg_requested_leverage": self._mean_value(all_rows, "requested_leverage", 0.0),
            "avg_effective_leverage_realized": self._mean_value(all_rows, "effective_leverage_realized", 0.0),
            "avg_hold_minutes": round(self._mean_value(effective_rows, "hold_seconds", 0.0) / 60.0, 6) if effective_rows else 0.0,
            "effective_learning_weight": round(effective_learning_weight, 6),
            "exploration_win_rate": round(exploration_wins / max(len(exploration_samples), 1), 6) if exploration_samples else 0.0,
            "top_setups": setup_summary,
            "market_regime_mix": self._top_counts(all_rows, "market_regime"),
            "market_phase_mix": self._top_counts(all_rows, "market_phase"),
            "breakout_phase_mix": self._top_counts(all_rows, "breakout_phase"),
            "trend_stage_mix": self._top_counts(all_rows, "trend_stage"),
            "market_phase_performance": self._performance_by_field(effective_rows, "market_phase"),
            "exploration_source_mix": self._top_counts(exploration_samples, "sample_source"),
            "leverage_bucket_performance": self._performance_by_bucket(effective_rows, "leverage", "leverage"),
            "margin_bucket_performance": self._performance_by_bucket(effective_rows, "margin_used", "margin"),
            "notional_bucket_performance": self._performance_by_bucket(effective_rows, "entry_notional_usdt", "notional"),
            "exit_followup_mix": self._top_counts(followups, "followup_verdict"),
        }

    def build_monthly_digest(self, review_key: str | None = None) -> Dict[str, Any]:
        all_rows = self.store.all_records()
        exploration_samples = self.store.recent_exploration(limit=10000)
        effective_rows = [row for row in all_rows if bool(row.get("count_in_learning", row.get("learning_tier") == "effective")) and bool(row.get("is_full_close", True))]
        review_id = review_key or datetime.now(ZoneInfo(settings.gpt_review_timezone)).date().isoformat()
        if not all_rows and not exploration_samples:
            return {
                "review_key": review_id,
                "trade_count": 0,
                "effective_trade_count": 0,
                "exploratory_trade_count": 0,
                "exploration_sample_count": 0,
                "top_setups": [],
                "market_phase_performance": [],
                "leverage_bucket_performance": [],
                "margin_bucket_performance": [],
                "notional_bucket_performance": [],
                "exit_followup_mix": [],
            }

        pnls = [float(r.get("pnl_net", r.get("pnl", 0.0)) or 0.0) for r in effective_rows]
        followups = [r for r in exploration_samples if str(r.get("sample_source", "") or "") == "post_exit_followup"]
        return {
            "review_key": review_id,
            "trade_count": len(effective_rows),
            "effective_trade_count": len(effective_rows),
            "exploratory_trade_count": len([r for r in all_rows if not bool(r.get("count_in_learning", False)) and bool(r.get("is_full_close", True))]),
            "exploration_sample_count": len(exploration_samples),
            "win_rate": round(len([p for p in pnls if p > 0]) / max(len(effective_rows), 1), 6) if effective_rows else 0.0,
            "avg_pnl": round(mean(pnls), 6) if effective_rows else 0.0,
            "total_pnl": round(sum(pnls), 6),
            "avg_pnl_on_margin_pct": self._mean_value(effective_rows, "pnl_on_margin_pct", 0.0),
            "avg_pnl_on_notional_bp": self._mean_value(effective_rows, "pnl_on_notional_bp", 0.0),
            "avg_entry_notional_usdt": self._mean_value(all_rows, "entry_notional_usdt", 0.0),
            "avg_margin_used": self._mean_value(all_rows, "margin_used", 0.0),
            "avg_leverage": self._mean_value(all_rows, "leverage", 0.0),
            "avg_effective_leverage_realized": self._mean_value(all_rows, "effective_leverage_realized", 0.0),
            "top_setups": self.store.setup_outcomes_summary(limit=10),
            "market_phase_mix": self._top_counts(all_rows, "market_phase", limit=8),
            "market_phase_performance": self._performance_by_field(effective_rows, "market_phase", limit=8),
            "leverage_bucket_performance": self._performance_by_bucket(effective_rows, "leverage", "leverage"),
            "margin_bucket_performance": self._performance_by_bucket(effective_rows, "margin_used", "margin"),
            "notional_bucket_performance": self._performance_by_bucket(effective_rows, "entry_notional_usdt", "notional"),
            "exploration_source_mix": self._top_counts(exploration_samples, "sample_source", limit=8),
            "exit_followup_mix": self._top_counts(followups, "followup_verdict", limit=6),
        }
