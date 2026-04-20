from __future__ import annotations

from datetime import date

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.models import AIAnalysisRecord, DailyReportRun, DiscordDeliveryLog, ScreeningCandidate


def serialize_report_run(report_run: DailyReportRun, delivery_logs: list[DiscordDeliveryLog]) -> dict:
    return {
        **report_run.__dict__,
        "delivery_logs": [log.__dict__ for log in delivery_logs],
    }


class DailyReportGenerator:
    def __init__(self, *, top_n: int, reason_max_length: int, risk_max_length: int) -> None:
        self.top_n = top_n
        self.reason_max_length = reason_max_length
        self.risk_max_length = risk_max_length

    def populate_report_run(
        self,
        session: Session,
        *,
        report_run: DailyReportRun,
        screening_run_id: int | None,
        report_date: date,
    ) -> DailyReportRun:
        report_run.report_date = report_date
        report_run.screening_run_id = screening_run_id
        report_run.top_n = self.top_n

        candidates = []
        if screening_run_id is not None:
            candidates = session.scalars(
                select(ScreeningCandidate)
                .where(
                    ScreeningCandidate.run_id == screening_run_id,
                    ScreeningCandidate.status == "ready",
                )
                .order_by(ScreeningCandidate.rank_position.asc(), ScreeningCandidate.symbol.asc())
                .limit(self.top_n)
            ).all()

        items = [self._build_item(session, candidate) for candidate in candidates]
        report_run.qualified_count = len(items)
        report_run.payload_json = {
            "report_date": report_date.isoformat(),
            "screening_run_id": screening_run_id,
            "qualified_count": len(items),
            "top_n": self.top_n,
            "items": items,
            "today_no_qualified_picks": len(items) == 0,
        }
        report_run.rendered_content = self._render_content(report_date=report_date, items=items)
        report_run.status = "prepared"
        report_run.error_detail = None
        return report_run

    def _build_item(self, session: Session, candidate: ScreeningCandidate) -> dict:
        return {
            "rank_position": candidate.rank_position,
            "symbol": candidate.symbol,
            "symbol_name": candidate.symbol_name,
            "overall_score": self._format_number(candidate.overall_score),
            "entry_zone": self._format_entry_zone(candidate.entry_zone_low, candidate.entry_zone_high),
            "stop_loss": self._format_number(candidate.stop_loss),
            "take_profit_1": self._format_number(candidate.take_profit_1),
            "take_profit_2": self._format_number(candidate.take_profit_2),
            "risk_reward_ratio": self._format_number(candidate.risk_reward_ratio),
            "reason": self._build_reason(session, candidate),
            "risk": self._build_risk(candidate),
        }

    def _build_reason(self, session: Session, candidate: ScreeningCandidate) -> str:
        analysis = session.scalar(
            select(AIAnalysisRecord)
            .where(
                AIAnalysisRecord.target_type == "screening_candidate",
                AIAnalysisRecord.target_id == candidate.id,
                AIAnalysisRecord.analysis_kind == "candidate_selection_reason",
            )
            .order_by(desc(AIAnalysisRecord.generated_at), desc(AIAnalysisRecord.id))
        )
        if analysis is not None and analysis.summary:
            summary = self._compact_text(analysis.summary, self.reason_max_length)
            if summary:
                return summary

        parts: list[str] = []
        pattern_label = ((candidate.evidence or {}).get("pattern") or {}).get("label")
        if pattern_label:
            parts.append(str(pattern_label))
        trend_score = (candidate.sub_scores or {}).get("trend")
        if trend_score is not None:
            parts.append(f"trend {float(trend_score):.1f}")
        momentum_score = (candidate.sub_scores or {}).get("momentum")
        if momentum_score is not None:
            parts.append(f"momentum {float(momentum_score):.1f}")
        if candidate.value_summary:
            parts.append(str(candidate.value_summary))

        if not parts:
            return "evidence insufficient"
        return self._compact_text(" | ".join(parts), self.reason_max_length)

    def _build_risk(self, candidate: ScreeningCandidate) -> str:
        reasons = list((candidate.risk_flags or {}).get("reasons", []))
        reasons.extend((candidate.value_risks or {}).get("reasons", []))
        ordered = []
        for reason in reasons:
            reason_text = str(reason).strip()
            if reason_text and reason_text not in ordered:
                ordered.append(reason_text)
        if not ordered:
            return "evidence insufficient"
        return self._compact_text(", ".join(ordered), self.risk_max_length)

    def _render_content(self, *, report_date: date, items: list[dict]) -> str:
        lines = [
            "台股短線每日推播",
            f"日期: {report_date.isoformat()}",
        ]
        if not items:
            lines.extend(
                [
                    "today no qualified picks",
                    "說明: 今日沒有符合規則門檻且具足夠真實資料支撐的股票。",
                ]
            )
            return "\n".join(lines)

        lines.append(f"今日前 {len(items)} 名")
        for item in items:
            lines.extend(
                [
                    (
                        f"{item['rank_position']}. {item['symbol']} {item['symbol_name'] or ''}".strip()
                        + f" | 分數 {item['overall_score']}"
                    ),
                    (
                        f"進場 {item['entry_zone']} | 止損 {item['stop_loss']} | "
                        f"止盈 {item['take_profit_1']} / {item['take_profit_2']} | "
                        f"風報比 {item['risk_reward_ratio']}"
                    ),
                    f"理由: {item['reason']}",
                    f"風險: {item['risk']}",
                ]
            )
        return "\n".join(lines)

    def _compact_text(self, text: str, max_length: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3].rstrip() + "..."

    def _format_entry_zone(self, low: float | None, high: float | None) -> str:
        low_text = self._format_number(low)
        high_text = self._format_number(high)
        if low_text == "unavailable" or high_text == "unavailable":
            return "unavailable"
        return f"{low_text}-{high_text}"

    def _format_number(self, value: float | None) -> str:
        if value is None:
            return "unavailable"
        return f"{float(value):.2f}"
