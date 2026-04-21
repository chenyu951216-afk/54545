from __future__ import annotations

from tw_stock_ai.ai_adapters.base import AIRequest
from tw_stock_ai.ai_adapters.openai_responses import OpenAIResponsesAdapter
from tw_stock_ai.config import Settings


def test_openai_responses_adapter_parses_official_response_shape(monkeypatch) -> None:
    def fake_http_post_json(url: str, payload: dict, *, headers=None, timeout=30):  # noqa: ANN001
        assert url == "https://api.openai.com/v1/responses"
        assert payload["model"] == "gpt-5-mini"
        assert headers["Authorization"] == "Bearer test-openai-key"
        return {
            "id": "resp_123",
            "model": "gpt-5-mini",
            "status": "completed",
            "output_text": "這是根據證據整理出的說明。",
            "usage": {"input_tokens": 111, "output_tokens": 22},
        }

    monkeypatch.setattr("tw_stock_ai.ai_adapters.openai_responses.http_post_json", fake_http_post_json)
    settings = Settings(
        ai_provider="openai",
        ai_model="gpt-5-mini",
        openai_api_key="test-openai-key",
        openai_base_url="https://api.openai.com/v1/responses",
    )
    adapter = OpenAIResponsesAdapter(settings)
    response = adapter.generate(
        AIRequest(
            prompt_name="candidate_selection_reason",
            prompt_text="請根據證據整理原因",
            evidence={"score": 90},
            max_output_tokens=200,
        )
    )

    assert response.provider == "openai"
    assert response.status == "completed"
    assert response.summary == "這是根據證據整理出的說明。"
    assert response.input_tokens == 111
    assert response.output_tokens == 22
