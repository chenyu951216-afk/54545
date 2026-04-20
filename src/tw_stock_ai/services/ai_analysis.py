from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.ai_adapters.base import AIRequest
from tw_stock_ai.config import get_settings
from tw_stock_ai.models import AIAnalysisRecord, Holding, ScreeningCandidate
from tw_stock_ai.schemas import AIAnalysisRead
from tw_stock_ai.services.ai_budget import within_budget
from tw_stock_ai.services.ai_guardrails import AIGuardrails
from tw_stock_ai.services.ai_registry import build_ai_adapter
from tw_stock_ai.services.ai_token_control import estimate_cost_twd, estimate_tokens, truncate_evidence
from tw_stock_ai.services.cost_control import CostControlService
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.logging_config import get_logger
from tw_stock_ai.services.prompt_loader import render_prompt
from tw_stock_ai.services.rate_limits import RateLimitExceededError, RateLimitService
from tw_stock_ai.services.usage_tracking import UsageTracker

logger = get_logger("tw_stock_ai.ai")


class AIAnalysisService:
    candidate_prompt_kinds = {
        "candidate_news_summary": "candidate_news_summary",
        "candidate_financial_highlights": "candidate_financial_highlights",
        "candidate_selection_reason": "candidate_selection_reason",
        "candidate_risk_summary": "candidate_risk_summary",
    }

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.adapter = build_ai_adapter(self.settings)
        self.guardrails = AIGuardrails(self.settings)
        self.flags = FeatureFlagService(self.settings)
        self.cost_control = CostControlService(self.settings)
        self.rate_limits = RateLimitService(self.settings)
        self.usage_tracker = UsageTracker()

    def analyze_top_candidates(self, session: Session, run_id: int) -> list[AIAnalysisRead]:
        if not self.guardrails.feature_allows(target_type="screening_candidate"):
            return []

        candidates = session.scalars(
            select(ScreeningCandidate)
            .where(
                ScreeningCandidate.run_id == run_id,
                ScreeningCandidate.status == "ready",
            )
            .order_by(ScreeningCandidate.rank_position.asc())
            .limit(self.settings.ai_top_n_candidates)
        ).all()

        results: list[AIAnalysisRead] = []
        for candidate in candidates:
            if not self.guardrails.candidate_symbol_allowed(candidate.symbol):
                continue
            results.extend(self._analyze_candidate(session, candidate))
        session.commit()
        return results

    def analyze_holding(self, session: Session, holding_id: int) -> list[AIAnalysisRead]:
        from tw_stock_ai.services.portfolio import enrich_holding

        if not self.guardrails.feature_allows(target_type="holding"):
            return []
        holding = session.get(Holding, holding_id)
        if holding is None:
            return []
        enriched = enrich_holding(session, holding)
        evidence = {
            "holding": {
                "symbol": enriched.symbol,
                "trend_status": enriched.trend_status,
                "exit_signal": enriched.exit_signal,
                "latest_close": enriched.latest_close,
                "unrealized_pnl": enriched.unrealized_pnl,
                "evidence": enriched.evidence,
            }
        }
        result = self._run_prompt(
            session=session,
            target_type="holding",
            target_id=holding.id,
            symbol=holding.symbol,
            analysis_kind="holding_trend_review",
            prompt_name="holding_trend_review",
            prompt_context={
                "symbol": holding.symbol,
                "evidence_json": evidence,
            },
            evidence=evidence,
        )
        session.commit()
        return [result]

    def latest_for_target(self, session: Session, *, target_type: str, target_id: int) -> list[AIAnalysisRead]:
        rows = session.scalars(
            select(AIAnalysisRecord)
            .where(
                AIAnalysisRecord.target_type == target_type,
                AIAnalysisRecord.target_id == target_id,
            )
            .order_by(AIAnalysisRecord.generated_at.desc(), AIAnalysisRecord.id.desc())
        ).all()

        latest_by_kind: dict[str, AIAnalysisRecord] = {}
        for row in rows:
            latest_by_kind.setdefault(row.analysis_kind, row)
        return [AIAnalysisRead.model_validate(item) for item in latest_by_kind.values()]

    def _analyze_candidate(self, session: Session, candidate: ScreeningCandidate) -> list[AIAnalysisRead]:
        evidence = {
            "technical": {
                "overall_score": candidate.overall_score,
                "sub_scores": candidate.sub_scores,
                "pattern_label": (candidate.evidence or {}).get("pattern", {}).get("label"),
                "evidence": candidate.evidence,
            },
            "value": {
                "value_score": candidate.value_score,
                "growth_score": candidate.growth_score,
                "quality_score": candidate.quality_score,
                "valuation_score": candidate.valuation_score,
                "catalyst_score": candidate.catalyst_score,
                "summary": candidate.value_summary,
                "evidence": candidate.treasure_evidence,
            },
            "matched_news": (candidate.treasure_evidence or {}).get("news", {}).get("matched_news", []),
            "fundamental": (candidate.treasure_evidence or {}).get("fundamental", {}),
            "risk_reasons": sorted(
                set(
                    list((candidate.risk_flags or {}).get("reasons", []))
                    + list((candidate.value_risks or {}).get("reasons", []))
                )
            ),
        }

        if not evidence["technical"]["evidence"]:
            evidence["insufficient"] = True
            evidence["reason"] = "candidate_evidence_missing"

        analyses: list[AIAnalysisRead] = []
        prompt_kinds = dict(self.candidate_prompt_kinds)
        if not self.settings.news_analysis_enabled:
            prompt_kinds.pop("candidate_news_summary", None)

        for analysis_kind, prompt_name in prompt_kinds.items():
            if not self.guardrails.candidate_prompt_allowed(prompt_name):
                continue
            analyses.append(
                self._run_prompt(
                    session=session,
                    target_type="screening_candidate",
                    target_id=candidate.id,
                    symbol=candidate.symbol,
                    analysis_kind=analysis_kind,
                    prompt_name=prompt_name,
                    prompt_context={
                        "symbol": candidate.symbol,
                        "symbol_name": candidate.symbol_name or "unknown",
                        "evidence_json": evidence,
                    },
                    evidence=evidence,
                )
            )
        return analyses

    def _run_prompt(
        self,
        *,
        session: Session,
        target_type: str,
        target_id: int,
        symbol: str | None,
        analysis_kind: str,
        prompt_name: str,
        prompt_context: dict,
        evidence: dict,
    ) -> AIAnalysisRead:
        trimmed_evidence = truncate_evidence(evidence, self.settings)
        if target_type == "holding" and not self.guardrails.holding_prompt_allowed(prompt_name):
            trimmed_evidence = {"insufficient": True, "reason": "holding_prompt_not_allowed"}

        cache_key = self.guardrails.build_cache_key(
            target_type=target_type,
            target_id=target_id,
            analysis_kind=analysis_kind,
            evidence=trimmed_evidence,
        )
        cached_record = self.guardrails.get_cached_analysis(session, cache_key=cache_key)
        if cached_record is not None:
            logger.info("ai_cache_hit target_type=%s target_id=%s analysis_kind=%s", target_type, target_id, analysis_kind)
            return AIAnalysisRead.model_validate(cached_record)

        prompt_text = render_prompt(prompt_name, {**prompt_context, "evidence_json": trimmed_evidence})
        input_tokens = estimate_tokens(prompt_text)
        estimated_cost = estimate_cost_twd(input_tokens, self.settings.ai_max_output_tokens, self.settings)
        operation = (
            "candidate_ai_analysis"
            if target_type == "screening_candidate"
            else "holding_ai_analysis"
        )

        try:
            if self.flags.is_enabled("cost_guardrails") and target_type == "screening_candidate":
                self.rate_limits.enforce(
                    session,
                    operation=operation,
                    limit=self.settings.rate_limit_candidate_ai_calls_per_window,
                )
            elif self.flags.is_enabled("cost_guardrails") and target_type == "holding":
                self.rate_limits.enforce(
                    session,
                    operation=operation,
                    limit=self.settings.rate_limit_holding_ai_calls_per_window,
                )
        except RateLimitExceededError as exc:
            response = self.adapter.generate(
                AIRequest(
                    prompt_name=prompt_name,
                    prompt_text="evidence insufficient",
                    evidence={"insufficient": True, "reason": str(exc)},
                    max_output_tokens=self.settings.ai_max_output_tokens,
                )
            )
            response.status = "rate_limited"
        else:
            if not within_budget(session, self.settings, estimated_cost):
                response = self.adapter.generate(
                    AIRequest(
                        prompt_name=prompt_name,
                        prompt_text="evidence insufficient",
                        evidence={"insufficient": True, "reason": "budget_exceeded"},
                        max_output_tokens=self.settings.ai_max_output_tokens,
                    )
                )
                response.status = "budget_blocked"
            elif self.flags.is_enabled("cost_guardrails") and not self.cost_control.within_overall_budget(
                session, estimated_cost
            ):
                response = self.adapter.generate(
                    AIRequest(
                        prompt_name=prompt_name,
                        prompt_text="evidence insufficient",
                        evidence={"insufficient": True, "reason": "overall_budget_exceeded"},
                        max_output_tokens=self.settings.ai_max_output_tokens,
                    )
                )
                response.status = "budget_blocked"
            elif not self.settings.ai_enabled and self.settings.ai_fallback_enabled:
                response = self.adapter.generate(
                    AIRequest(
                        prompt_name=prompt_name,
                        prompt_text=prompt_text,
                        evidence=trimmed_evidence,
                        max_output_tokens=self.settings.ai_max_output_tokens,
                    )
                )
            elif not self.settings.ai_enabled:
                response = self.adapter.generate(
                    AIRequest(
                        prompt_name=prompt_name,
                        prompt_text="evidence insufficient",
                        evidence={"insufficient": True, "reason": "ai_disabled"},
                        max_output_tokens=self.settings.ai_max_output_tokens,
                    )
                )
                response.status = "disabled"
            else:
                response = self.adapter.generate(
                    AIRequest(
                        prompt_name=prompt_name,
                        prompt_text=prompt_text,
                        evidence=trimmed_evidence,
                        max_output_tokens=self.settings.ai_max_output_tokens,
                    )
                )

        self.rate_limits.record(
            session,
            operation=operation,
            status=response.status,
            metadata={"target_type": target_type, "target_id": target_id, "prompt_name": prompt_name},
        )
        self.usage_tracker.record(
            session,
            event_type="ai_call",
            operation=operation,
            provider=response.provider,
            status=response.status,
            estimated_cost_twd=response.estimated_cost_twd,
            metadata={"symbol": symbol, "prompt_name": prompt_name},
        )

        record = AIAnalysisRecord(
            target_type=target_type,
            target_id=target_id,
            symbol=symbol,
            analysis_kind=analysis_kind,
            prompt_name=prompt_name,
            provider=response.provider,
            model_name=response.model_name,
            status=response.status,
            summary=response.summary,
            details=response.details,
            evidence_snapshot=trimmed_evidence,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_twd=response.estimated_cost_twd,
            fallback_used=response.fallback_used,
            cache_key=cache_key,
            generated_at=response.generated_at,
        )
        session.add(record)
        session.flush()
        logger.info(
            "ai_analysis_recorded target_type=%s target_id=%s analysis_kind=%s status=%s provider=%s",
            target_type,
            target_id,
            analysis_kind,
            response.status,
            response.provider,
        )
        return AIAnalysisRead.model_validate(record)
