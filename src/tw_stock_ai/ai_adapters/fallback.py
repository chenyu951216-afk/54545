from __future__ import annotations

from tw_stock_ai.ai_adapters.base import AIRequest, AIResponse, BaseAIAdapter


class FallbackAIAdapter(BaseAIAdapter):
    provider_name = "fallback"
    model_name = "fallback-v1"

    def generate(self, request: AIRequest) -> AIResponse:
        evidence = request.evidence or {}
        status = "completed"

        if evidence.get("insufficient"):
            summary = "evidence insufficient"
            details = {"reason": evidence.get("reason", "insufficient_evidence")}
        elif request.prompt_name == "candidate_news_summary":
            matched_news = evidence.get("matched_news", [])
            if not matched_news:
                summary = "evidence insufficient"
                details = {"reason": "no_news_evidence"}
            else:
                titles = "；".join(item.get("title", "") for item in matched_news[:3])
                summary = f"近期新聞重點：{titles}"
                details = {"news_count": len(matched_news)}
        elif request.prompt_name == "candidate_financial_highlights":
            fundamentals = evidence.get("fundamental", {})
            if not fundamentals or not any(value is not None for value in fundamentals.values()):
                summary = "evidence insufficient"
                details = {"reason": "no_financial_evidence"}
            else:
                summary = (
                    f"財報重點：EPS {fundamentals.get('eps')}、ROE {fundamentals.get('roe')}、"
                    f"毛利率 {fundamentals.get('gross_margin')}、營益率 {fundamentals.get('operating_margin')}"
                )
                details = {"fields_used": ["eps", "roe", "gross_margin", "operating_margin"]}
        elif request.prompt_name == "candidate_selection_reason":
            technical = evidence.get("technical", {})
            value = evidence.get("value", {})
            if not technical and not value:
                summary = "evidence insufficient"
                details = {"reason": "no_selection_evidence"}
            else:
                summary = (
                    f"入選原因：短線分數 {technical.get('overall_score')}，"
                    f"趨勢 {technical.get('sub_scores', {}).get('trend_score')}，"
                    f"型態 {technical.get('pattern_label')}，"
                    f"寶藏分 {value.get('value_score')}"
                )
                details = {"technical": technical, "value": value}
        elif request.prompt_name == "candidate_risk_summary":
            risks = evidence.get("risk_reasons", [])
            summary = "風險整理：" + ("、".join(risks) if risks else "未偵測到額外規則風險")
            details = {"risk_reasons": risks}
        elif request.prompt_name == "holding_trend_review":
            holding = evidence.get("holding", {})
            if not holding:
                summary = "evidence insufficient"
                details = {"reason": "no_holding_evidence"}
            else:
                summary = (
                    f"持股趨勢檢查：目前趨勢 {holding.get('trend_status')}，"
                    f"出場訊號 {holding.get('exit_signal')}，"
                    f"最新價 {holding.get('latest_close')}。"
                )
                details = {"holding": holding}
        else:
            status = "insufficient_evidence"
            summary = "evidence insufficient"
            details = {"reason": "unknown_prompt"}

        input_tokens = max(len(request.prompt_text) // 4, 1)
        output_tokens = max(len(summary) // 4, 1)
        return AIResponse(
            provider=self.provider_name,
            model_name=self.model_name,
            status=status,
            summary=summary,
            details=details,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_twd=0.0,
            fallback_used=True,
        )
