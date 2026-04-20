from __future__ import annotations

from typing import Any, Dict, List


class AIReviewJudgeService:
    def evaluate(self, digest: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
        recs: List[Dict[str, Any]] = list(review.get("recommendations", []) or [])
        win_rate = float(digest.get("win_rate", 0.0) or 0.0)
        avg_pnl = float(digest.get("avg_pnl", 0.0) or 0.0)
        losses = int(digest.get("losses", 0) or 0)
        trades = int(digest.get("trade_count", 0) or 0)
        drift_alert = digest.get("drift_alert", {}) if isinstance(digest.get("drift_alert"), dict) else {}
        drift_severity = str(drift_alert.get("severity", "none") or "none")
        drift_score = float(drift_alert.get("score", 0.0) or 0.0)
        objections: List[Dict[str, Any]] = []
        accepted: List[Dict[str, Any]] = []
        needs_more = False

        for rec in recs:
            area = str(rec.get("area", "general"))
            direction = str(rec.get("direction", "hold"))
            confidence = float(rec.get("confidence", 0.5) or 0.5)
            reason = str(rec.get("reason", ""))

            if trades < 4 and direction in {"tighten", "decrease"}:
                objections.append({
                    "recommendation": rec,
                    "objection": "sample_too_small_for_restriction",
                    "evidence": {"trade_count": trades},
                })
                needs_more = True
                continue
            if win_rate >= 0.58 and avg_pnl > 0 and direction in {"tighten", "decrease"} and confidence < 0.75:
                objections.append({
                    "recommendation": rec,
                    "objection": "recent_results_not_supporting_defensive_shift",
                    "evidence": {"win_rate": win_rate, "avg_pnl": avg_pnl, "area": area, "reason": reason},
                })
                needs_more = True
                continue
            if losses >= max(3, trades // 2) and direction in {"increase", "loosen"} and confidence < 0.82:
                objections.append({
                    "recommendation": rec,
                    "objection": "loss_cluster_requires_stronger_evidence_before_more_aggression",
                    "evidence": {"losses": losses, "trade_count": trades, "area": area},
                })
                needs_more = True
                continue
            if drift_severity in {"high", "critical"} and direction in {"increase", "loosen"}:
                objections.append({
                    "recommendation": rec,
                    "objection": "learning_drift_requires_defensive_bias",
                    "evidence": {"drift_severity": drift_severity, "drift_score": drift_score, "area": area},
                })
                needs_more = True
                continue
            if drift_severity == "medium" and area in {"entry", "sizing", "leverage"} and direction in {"increase", "loosen"} and confidence < 0.86:
                objections.append({
                    "recommendation": rec,
                    "objection": "guarded_mode_blocks_aggressive_shift_without_strong_evidence",
                    "evidence": {"drift_severity": drift_severity, "drift_score": drift_score, "area": area},
                })
                needs_more = True
                continue
            accepted.append(rec)

        verdict = "accept" if not objections else ("partial_accept" if accepted else "discuss_more")
        return {
            "verdict": verdict,
            "accepted_recommendations": accepted,
            "objections": objections,
            "needs_more_discussion": needs_more,
        }
