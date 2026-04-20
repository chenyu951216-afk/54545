from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class AIRequest:
    prompt_name: str
    prompt_text: str
    evidence: dict[str, Any]
    max_output_tokens: int
    temperature: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AIResponse:
    provider: str
    model_name: str
    status: str
    summary: str
    details: dict[str, Any]
    input_tokens: int
    output_tokens: int
    estimated_cost_twd: float
    fallback_used: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAIAdapter(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(self, request: AIRequest) -> AIResponse:
        raise NotImplementedError
