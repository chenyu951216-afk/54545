from __future__ import annotations

from tw_stock_ai.adapters.http_utils import HttpFetchError, http_post_json
from tw_stock_ai.ai_adapters.base import AIRequest, AIResponse, BaseAIAdapter
from tw_stock_ai.config import Settings, get_settings


class OpenAIResponsesAdapter(BaseAIAdapter):
    provider_name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.ai_model

    def generate(self, request: AIRequest) -> AIResponse:
        if not self.settings.openai_api_key:
            return AIResponse(
                provider=self.provider_name,
                model_name=self.model_name,
                status="unavailable",
                summary="evidence insufficient",
                details={"reason": "openai_api_key_missing"},
                input_tokens=0,
                output_tokens=0,
                estimated_cost_twd=0.0,
                fallback_used=False,
            )

        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        if self.settings.openai_organization:
            headers["OpenAI-Organization"] = self.settings.openai_organization
        if self.settings.openai_project:
            headers["OpenAI-Project"] = self.settings.openai_project

        payload = {
            "model": self.settings.ai_model,
            "input": request.prompt_text,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        try:
            response_json = http_post_json(
                self.settings.openai_base_url,
                payload,
                headers=headers,
                timeout=self.settings.ai_timeout_seconds,
            )
        except HttpFetchError as exc:
            return AIResponse(
                provider=self.provider_name,
                model_name=self.model_name,
                status="failed",
                summary="evidence insufficient",
                details={"reason": str(exc)},
                input_tokens=0,
                output_tokens=0,
                estimated_cost_twd=0.0,
                fallback_used=False,
            )

        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        output_text = self._extract_output_text(response_json) or "evidence insufficient"
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return AIResponse(
            provider=self.provider_name,
            model_name=response_json.get("model", self.model_name),
            status="completed",
            summary=output_text,
            details={
                "response_id": response_json.get("id"),
                "provider_payload": {
                    "status": response_json.get("status"),
                    "incomplete_details": response_json.get("incomplete_details"),
                },
            },
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_twd=(
                input_tokens / 1000 * self.settings.ai_estimated_input_cost_per_1k_tokens_twd
                + output_tokens / 1000 * self.settings.ai_estimated_output_cost_per_1k_tokens_twd
            ),
            fallback_used=False,
        )

    @staticmethod
    def _extract_output_text(response_json: dict) -> str:
        if isinstance(response_json.get("output_text"), str) and response_json["output_text"].strip():
            return response_json["output_text"].strip()
        output = response_json.get("output")
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
        return "\n".join(parts).strip()
