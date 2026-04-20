from __future__ import annotations

import json

from tw_stock_ai.config import Settings
from tw_stock_ai.services.serialization import to_jsonable


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def truncate_evidence(evidence: dict, settings: Settings) -> dict:
    serialized = json.dumps(to_jsonable(evidence), ensure_ascii=False)
    if len(serialized) <= settings.ai_max_input_chars:
        return to_jsonable(evidence)

    trimmed = dict(to_jsonable(evidence))
    trimmed["truncated"] = True
    trimmed["truncation_reason"] = "ai_max_input_chars_exceeded"
    trimmed["evidence_preview"] = serialized[: settings.ai_max_input_chars]
    return trimmed


def estimate_cost_twd(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    return round(
        (input_tokens / 1000) * settings.ai_estimated_input_cost_per_1k_tokens_twd
        + (output_tokens / 1000) * settings.ai_estimated_output_cost_per_1k_tokens_twd,
        6,
    )
