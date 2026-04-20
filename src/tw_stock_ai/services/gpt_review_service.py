from __future__ import annotations

from typing import Any, Dict


class GPTReviewService:
    """
    Legacy path disabled on purpose.
    Keep this stub so old imports do not crash,
    but do not send any data to GPT.
    """

    def __init__(self) -> None:
        pass

    def available(self) -> bool:
        return False

    def deliberate(self, digest: Dict[str, Any], current_policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": "legacy_gpt_review_disabled",
            "suggestions": [],
            "discussion_log": [],
            "rounds_used": 0,
            "merged_patch": {},
        }
