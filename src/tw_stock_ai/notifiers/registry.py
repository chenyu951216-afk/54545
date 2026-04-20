from __future__ import annotations

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.notifiers.base import BaseNotifier
from tw_stock_ai.notifiers.discord_notifier import DiscordNotifier


def build_default_notifier(settings: Settings | None = None) -> BaseNotifier:
    effective = settings or get_settings()
    return DiscordNotifier(settings=effective)
