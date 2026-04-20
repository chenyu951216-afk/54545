from __future__ import annotations

from tw_stock_ai.ai_adapters.base import BaseAIAdapter
from tw_stock_ai.ai_adapters.fallback import FallbackAIAdapter
from tw_stock_ai.ai_adapters.openai_responses import OpenAIResponsesAdapter
from tw_stock_ai.ai_adapters.unavailable import UnavailableAIAdapter
from tw_stock_ai.config import Settings


def build_ai_adapter(settings: Settings) -> BaseAIAdapter:
    if settings.ai_provider == "openai":
        if settings.openai_api_key:
            return OpenAIResponsesAdapter(settings)
        if settings.ai_fallback_enabled:
            return FallbackAIAdapter()
        return UnavailableAIAdapter()
    if settings.ai_provider == "fallback":
        return FallbackAIAdapter()
    if settings.ai_fallback_enabled:
        return FallbackAIAdapter()
    return UnavailableAIAdapter()
