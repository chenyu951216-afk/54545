from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.models import AIAnalysisRecord
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.serialization import to_jsonable


class AIGuardrails:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.flags = FeatureFlagService(self.settings)

    def candidate_prompt_allowed(self, prompt_name: str) -> bool:
        allowed = {item.strip() for item in self.settings.ai_allowed_candidate_prompt_names.split(",") if item.strip()}
        return prompt_name in allowed

    def holding_prompt_allowed(self, prompt_name: str) -> bool:
        allowed = {item.strip() for item in self.settings.ai_allowed_holding_prompt_names.split(",") if item.strip()}
        return prompt_name in allowed

    def candidate_symbol_allowed(self, symbol: str) -> bool:
        allowlist = {item.strip() for item in self.settings.ai_candidate_symbol_allowlist.split(",") if item.strip()}
        if not allowlist:
            return True
        return symbol in allowlist

    def feature_allows(self, *, target_type: str) -> bool:
        if target_type == "screening_candidate":
            return self.flags.is_enabled("candidate_ai_analysis")
        if target_type == "holding":
            return self.flags.is_enabled("holding_ai_analysis")
        return False

    def build_cache_key(self, *, target_type: str, target_id: int, analysis_kind: str, evidence: dict) -> str:
        normalized = json.dumps(to_jsonable(evidence), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return f"{target_type}:{target_id}:{analysis_kind}:{digest}"

    def get_cached_analysis(self, session: Session, *, cache_key: str) -> AIAnalysisRecord | None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.settings.ai_cache_ttl_hours)
        return session.scalar(
            select(AIAnalysisRecord)
            .where(
                AIAnalysisRecord.cache_key == cache_key,
                AIAnalysisRecord.generated_at >= cutoff,
            )
            .order_by(desc(AIAnalysisRecord.generated_at), desc(AIAnalysisRecord.id))
        )
