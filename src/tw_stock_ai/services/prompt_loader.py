from __future__ import annotations

import json
from pathlib import Path
from string import Template

from tw_stock_ai.services.serialization import to_jsonable

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(prompt_name: str) -> str:
    path = PROMPTS_DIR / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


def render_prompt(prompt_name: str, context: dict) -> str:
    template = Template(load_prompt_template(prompt_name))
    normalized_context = {
        key: (
            json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in context.items()
    }
    return template.safe_substitute(normalized_context)
