from __future__ import annotations

from tw_stock_ai.ai_adapters.base import AIRequest, AIResponse, BaseAIAdapter


class UnavailableAIAdapter(BaseAIAdapter):
    provider_name = "unavailable"
    model_name = "unavailable"

    def generate(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            provider=self.provider_name,
            model_name=self.model_name,
            status="unavailable",
            summary="evidence insufficient",
            details={"reason": "ai_provider_not_configured"},
            input_tokens=0,
            output_tokens=0,
            estimated_cost_twd=0.0,
            fallback_used=True,
        )
