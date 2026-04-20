from __future__ import annotations

from tw_stock_ai.services.short_term_types import CandidateEvaluation


def rank_candidates(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return sorted(
        candidates,
        key=lambda item: (
            item.status == "ready",
            item.overall_score or -1.0,
            item.risk_reward_ratio or -1.0,
            item.evidence.get("adx", -1.0),
            item.symbol,
        ),
        reverse=True,
    )
