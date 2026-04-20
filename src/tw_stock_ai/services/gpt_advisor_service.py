from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from openai import OpenAI

from config.settings import settings


class GPTAdvisorService:
    """
    Cost-controlled GPT service.

    Rules:
    - live control is always disabled
    - only one compact daily review is allowed in the 00:00-00:05 window
    - review target is the previous local trading day
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def available(self) -> bool:
        return bool(self.client)

    def live_available(self) -> bool:
        return False

    def advise_live(self, candidate: Dict[str, Any], account_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        result = dict(candidate or {})
        result["gpt_live_used"] = False
        result["gpt_live_mode"] = "disabled_daily_only"
        return result

    def overlay_live(self, candidate: Dict[str, Any], account_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.advise_live(candidate, account_summary)

    def advise_live_trade_candidate(self, candidate: Dict[str, Any], current_policy: Dict[str, Any], account_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "enabled": False,
            "action": "hold",
            "reason": "live_gpt_disabled_daily_only",
        }

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(settings.gpt_review_timezone)

    def _now_local(self) -> datetime:
        return datetime.now(self._tz())

    def _next_midnight(self, now: datetime | None = None) -> datetime:
        local_now = now or self._now_local()
        next_day = (local_now + timedelta(days=1)).date()
        return datetime.combine(next_day, time(0, 0), tzinfo=self._tz())

    def in_daily_review_window(self, now: datetime | None = None) -> bool:
        local_now = now or self._now_local()
        return local_now.hour == 0 and local_now.minute < 5

    def target_review_day(self, now: datetime | None = None) -> str:
        local_now = now or self._now_local()
        return (local_now.date() - timedelta(days=1)).isoformat()

    def in_monthly_review_window(self, now: datetime | None = None) -> bool:
        local_now = now or self._now_local()
        return (
            settings.enable_monthly_gpt_review
            and local_now.day == settings.monthly_review_day
            and local_now.hour == 0
            and local_now.minute < 10
        )

    def target_monthly_review_key(self, now: datetime | None = None) -> str:
        local_now = now or self._now_local()
        return local_now.strftime("%Y-%m")

    def status(self) -> Dict[str, Any]:
        now = self._now_local()
        return {
            "available": self.available(),
            "live_available": False,
            "mode": "daily_only",
            "configured": bool(settings.openai_api_key),
            "next_run_local": self._next_midnight(now).isoformat(),
            "review_window": "00:00-00:05",
            "target_review_day": self.target_review_day(now),
            "monthly_review_enabled": settings.enable_monthly_gpt_review,
            "monthly_review_day": settings.monthly_review_day,
            "target_monthly_review_key": self.target_monthly_review_key(now),
        }

    def _response_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text).strip()
        try:
            return str(response.output[0].content[0].text).strip()
        except Exception:
            return ""

    def _create_response(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            return ""
        try:
            response = self.client.responses.create(
                model=settings.gpt_model,
                reasoning={"effort": settings.gpt_reasoning_effort},
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=settings.gpt_timeout_sec,
            )
            return self._response_text(response)
        except Exception:
            return ""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = candidate.replace("json", "", 1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except Exception:
                    return {}
        return {}

    def _compact_policy_summary(self, current_policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: current_policy.get(key)
            for key in [
                "protection_profile",
                "exit_style",
                "position_management_profile",
                "entry_confidence_shift",
                "size_multiplier_bias",
                "leverage_bias",
                "breakout_bias",
                "trend_follow_bias",
                "autonomy_level",
                "autonomy_state",
                "autonomy_limits",
                "autonomy_maturity_score",
                "drift_alert",
                "drift_score",
                "lesson_memory",
            ]
        }

    def review_daily_trades(self, digest: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        recommendations: List[Dict[str, Any]] = []
        win_rate = float(digest.get("win_rate", 0.0) or 0.0)
        avg_pnl = float(digest.get("avg_pnl", 0.0) or 0.0)
        exploration_win_rate = float(digest.get("exploration_win_rate", 0.0) or 0.0)
        exploratory_trade_count = int(digest.get("exploratory_trade_count", 0) or 0)
        effective_trade_count = int(digest.get("effective_trade_count", digest.get("trade_count", 0)) or 0)
        drift_alert = digest.get("drift_alert", {}) if isinstance(digest.get("drift_alert"), dict) else {}
        drift_severity = str(drift_alert.get("severity", "none") or "none")
        if win_rate < 0.45 and avg_pnl <= 0:
            recommendations.append({"area": "entry", "direction": "tighten", "confidence": 0.66, "reason": "low_win_rate_negative_avg_pnl"})
            recommendations.append({"area": "sizing", "direction": "decrease", "confidence": 0.64, "reason": "protect_small_capital_during_drawdown"})
        elif win_rate > 0.58 and avg_pnl > 0:
            recommendations.append({"area": "protection", "direction": "loosen", "confidence": 0.58, "reason": "allow_winners_more_room"})
        if exploratory_trade_count > effective_trade_count * 1.5 and exploration_win_rate < 0.45:
            recommendations.append({"area": "entry", "direction": "tighten", "confidence": 0.62, "reason": "exploration_overweight_without_edge"})
        elif exploratory_trade_count > 0 and exploration_win_rate >= 0.55:
            recommendations.append({"area": "entry", "direction": "hold", "confidence": 0.56, "reason": "exploration_showing_positive_edge_keep_collecting"})
        if drift_severity in {"medium", "high", "critical"}:
            recommendations.append({"area": "entry", "direction": "tighten", "confidence": 0.78, "reason": f"drift_alert_{drift_severity}"})
            recommendations.append({"area": "sizing", "direction": "decrease", "confidence": 0.76, "reason": "preserve_capital_during_relearning"})
        if drift_severity in {"high", "critical"}:
            recommendations.append({"area": "protection", "direction": "tighten", "confidence": 0.74, "reason": "drift_guard_needs_faster_damage_control"})

        summary = (
            f"day={digest.get('day')} trades={digest.get('trade_count', 0)} "
            f"win_rate={win_rate:.2%} avg_pnl={avg_pnl:.4f} total_pnl={float(digest.get('total_pnl', 0.0) or 0.0):.4f}"
        )
        if not self.client:
            return {
                "summary": f"rule_based_daily_review {summary}",
                "recommendations": recommendations,
                "discussion_log": [],
                "rounds_used": 0,
            }

        prompt = (
            "You are optimizing an automated crypto futures system. "
            "Use only the provided compact digest, learned memory summary, and current policy summary. "
            "Return strict JSON with keys: summary, recommendations. "
            "Each recommendation item must have area, direction, confidence, reason. "
            "Keep changes conservative for small capital and do not suggest live per-scan GPT control. "
            "If there is learning drift, prioritize defensive stabilization and short lesson-oriented guidance.\n"
            f"digest={json.dumps(digest, ensure_ascii=False)}\n"
            f"learned_memory_summary={json.dumps(current_policy.get('learning_memory_summary', {}), ensure_ascii=False)}\n"
            f"current_policy_summary={json.dumps(self._compact_policy_summary(current_policy), ensure_ascii=False)}"
        )
        text = self._create_response(
            "You are a cautious trading system reviewer that outputs compact JSON only.",
            prompt,
        )
        parsed = self._extract_json(text)
        return {
            "summary": str(parsed.get("summary") or f"gpt_daily_review {summary}"),
            "recommendations": list(parsed.get("recommendations") or recommendations),
            "discussion_log": [],
            "rounds_used": 1 if text else 0,
        }

    def discuss_disagreement(
        self,
        digest: Dict[str, Any],
        review: Dict[str, Any],
        judge_feedback: Dict[str, Any],
        current_policy: Dict[str, Any],
        round_index: int,
    ) -> Dict[str, Any]:
        if not self.client:
            return {
                "consensus_summary": str(review.get("summary", "")),
                "updated_recommendations": list(review.get("recommendations", [])),
                "needs_more_discussion": False,
                "round": round_index,
            }
        prompt = (
            "Given the prior review and the judge objections, produce a tighter conservative revision. "
            "Return strict JSON with keys: consensus_summary, updated_recommendations, needs_more_discussion.\n"
            f"digest={json.dumps(digest, ensure_ascii=False)}\n"
            f"review={json.dumps(review, ensure_ascii=False)}\n"
            f"judge_feedback={json.dumps(judge_feedback, ensure_ascii=False)}\n"
            f"current_policy_summary={json.dumps(self._compact_policy_summary(current_policy), ensure_ascii=False)}"
        )
        text = self._create_response(
            "You are a conservative reviewer resolving disagreements for a trading system. Output JSON only.",
            prompt,
        )
        parsed = self._extract_json(text)
        return {
            "consensus_summary": str(parsed.get("consensus_summary") or review.get("summary", "")),
            "updated_recommendations": list(parsed.get("updated_recommendations") or review.get("recommendations", [])),
            "needs_more_discussion": bool(parsed.get("needs_more_discussion", False)),
            "round": round_index,
        }

    def recommend_adjustments(self, digest: Dict[str, Any], consensus: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        fallback: Dict[str, Any] = {
            "summary": str(consensus.get("consensus_summary", "")),
            "trade_count": int(digest.get("trade_count", 0) or 0),
            "win_rate": float(digest.get("win_rate", 0.0) or 0.0),
            "avg_pnl": float(digest.get("avg_pnl", 0.0) or 0.0),
            "effective_trade_count": int(digest.get("effective_trade_count", digest.get("trade_count", 0)) or 0),
            "exploratory_trade_count": int(digest.get("exploratory_trade_count", 0) or 0),
            "exploration_win_rate": float(digest.get("exploration_win_rate", 0.0) or 0.0),
        }
        if float(digest.get("win_rate", 0.0) or 0.0) < 0.45:
            fallback.update({
                "entry_confidence_shift": min(float(current_policy.get("entry_confidence_shift", 0.0) or 0.0) + 0.012, 0.08),
                "size_multiplier_bias": max(float(current_policy.get("size_multiplier_bias", 1.0) or 1.0) - 0.05, settings.adaptive_size_floor),
                "position_management_profile": "defensive",
                "protection_profile": "tight",
            })
        elif float(digest.get("win_rate", 0.0) or 0.0) > 0.58 and float(digest.get("avg_pnl", 0.0) or 0.0) > 0:
            fallback.update({
                "entry_confidence_shift": max(float(current_policy.get("entry_confidence_shift", 0.0) or 0.0) - 0.006, -0.08),
                "size_multiplier_bias": min(float(current_policy.get("size_multiplier_bias", 1.0) or 1.0) + 0.03, settings.adaptive_size_ceiling),
                "position_management_profile": "press_winners",
                "protection_profile": "balanced",
            })
        drift_alert = current_policy.get("drift_alert", {}) if isinstance(current_policy.get("drift_alert"), dict) else {}
        drift_severity = str(drift_alert.get("severity", "none") or "none")
        if drift_severity in {"medium", "high", "critical"}:
            fallback.update({
                "entry_confidence_shift": min(max(float(fallback.get("entry_confidence_shift", current_policy.get("entry_confidence_shift", 0.0)) or 0.0), float(current_policy.get("entry_confidence_shift", 0.0) or 0.0)) + 0.006, 0.08),
                "size_multiplier_bias": min(float(fallback.get("size_multiplier_bias", current_policy.get("size_multiplier_bias", 1.0)) or 1.0), 0.95 if drift_severity == "medium" else 0.88 if drift_severity == "high" else 0.82),
                "leverage_bias": min(float(current_policy.get("leverage_bias", 1.0) or 1.0), 0.96 if drift_severity == "medium" else 0.90 if drift_severity == "high" else 0.84),
                "position_management_profile": "defensive",
                "protection_profile": "tight" if drift_severity in {"high", "critical"} else fallback.get("protection_profile", current_policy.get("protection_profile", "balanced")),
                "summary": f"{fallback.get('summary', '')} drift={drift_severity}".strip(),
            })

        if not self.client:
            return fallback

        prompt = (
            "Return strict JSON only. "
            "Suggest bounded daily parameter adjustments for a small-cap automated futures system. "
            "Do not enable live GPT control. "
            "If drift_alert is medium/high/critical, prefer stabilization, clearer filters, smaller size, and lesson retention over aggression. "
            "Allowed keys include summary, entry_confidence_shift, size_multiplier_bias, leverage_bias, "
            "protection_profile, exit_style, position_management_profile, break_even_trigger_rr, trailing_activation_rr, "
            "trailing_buffer_atr, partial_take_profit_rr, break_even_lock_ratio, trailing_step_rr, tp1_fraction, tp2_fraction.\n"
            f"digest={json.dumps(digest, ensure_ascii=False)}\n"
            f"consensus={json.dumps(consensus, ensure_ascii=False)}\n"
            f"learned_memory_summary={json.dumps(current_policy.get('learning_memory_summary', {}), ensure_ascii=False)}\n"
            f"current_policy_summary={json.dumps(self._compact_policy_summary(current_policy), ensure_ascii=False)}"
        )
        text = self._create_response(
            "You are a conservative parameter optimizer. Output compact JSON only.",
            prompt,
        )
        parsed = self._extract_json(text)
        return {**fallback, **parsed}

    def review_monthly_progress(self, digest: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        summary = (
            f"month={digest.get('review_key')} trades={digest.get('trade_count', 0)} "
            f"win_rate={float(digest.get('win_rate', 0.0) or 0.0):.2%} total_pnl={float(digest.get('total_pnl', 0.0) or 0.0):.4f} "
            f"avg_margin_return={float(digest.get('avg_pnl_on_margin_pct', 0.0) or 0.0):.2f}%"
        )
        fallback = {
            "summary": f"rule_based_monthly_review {summary}",
            "strengths": [f"top_setups={json.dumps(digest.get('top_setups', [])[:3], ensure_ascii=False)}"],
            "risks": [f"exit_followup_mix={json.dumps(digest.get('exit_followup_mix', [])[:3], ensure_ascii=False)}"],
            "next_focus": [
                f"market_phase_performance={json.dumps(digest.get('market_phase_performance', [])[:3], ensure_ascii=False)}",
                f"notional_bucket_performance={json.dumps(digest.get('notional_bucket_performance', [])[:3], ensure_ascii=False)}",
            ],
        }
        if not self.client:
            return fallback

        prompt = (
            "You are preparing a monthly compact review for a small-cap automated crypto futures system. "
            "Return strict JSON with keys: summary, strengths, risks, next_focus. "
            "Each list should contain at most 3 short items. "
            "Do not produce long analysis. Do not enable live GPT control. "
            "Reflect ongoing lesson memory and whether recent drift should change the next focus.\n"
            f"monthly_digest={json.dumps(digest, ensure_ascii=False)}\n"
            f"learned_memory_summary={json.dumps(current_policy.get('learning_memory_summary', {}), ensure_ascii=False)}\n"
            f"current_policy_summary={json.dumps(self._compact_policy_summary(current_policy), ensure_ascii=False)}"
        )
        text = self._create_response(
            "You are a compact monthly trading system reviewer. Output JSON only and keep it short.",
            prompt,
        )
        parsed = self._extract_json(text)
        return {
            "summary": str(parsed.get("summary") or fallback["summary"]),
            "strengths": list(parsed.get("strengths") or fallback["strengths"]),
            "risks": list(parsed.get("risks") or fallback["risks"]),
            "next_focus": list(parsed.get("next_focus") or fallback["next_focus"]),
        }
