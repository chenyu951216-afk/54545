from __future__ import annotations

import signal
import time

from tw_stock_ai.config import get_settings
from tw_stock_ai.db import init_db
from tw_stock_ai.services.jobs import build_scheduler, maybe_run_startup_bootstrap, maybe_run_startup_catchup
from tw_stock_ai.services.logging_config import configure_logging, get_logger

logger = get_logger("tw_stock_ai.worker")


def main() -> None:
    configure_logging("worker")
    settings = get_settings()
    init_db()

    maybe_run_startup_bootstrap()
    catchup_summary = maybe_run_startup_catchup()

    scheduler = build_scheduler()
    scheduler.start()

    logger.info(
        "worker_started settings=%s",
        {
            "scheduler_timezone": settings.scheduler_timezone,
            "screening_hour": settings.screening_hour,
            "screening_minute": settings.screening_minute,
            "screening_weekdays": settings.screening_weekdays,
            "news_polling_enabled": settings.news_polling_enabled,
            "news_poll_interval_minutes": settings.news_poll_interval_minutes,
            "startup_catchup": catchup_summary,
        },
    )

    should_run = True

    def _handle_signal(signum, frame) -> None:  # type: ignore[unused-argument]
        nonlocal should_run
        should_run = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while should_run:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
