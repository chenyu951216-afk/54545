from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tw_stock_ai.config import get_settings
from tw_stock_ai.db import init_db
from tw_stock_ai.routers.api import router as api_router
from tw_stock_ai.routers.ui import router as ui_router
from tw_stock_ai.services.jobs import build_scheduler
from tw_stock_ai.services.logging_config import configure_logging, get_logger
from tw_stock_ai.services.request_logging import RequestLoggingMiddleware

scheduler = None
logger = get_logger("tw_stock_ai.main")
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler

    configure_logging("web")
    settings = get_settings()
    init_db()
    logger.info(
        "web_startup app_env=%s configured_port=%s runtime_port_env=%s scheduler_enabled=%s static_dir=%s",
        settings.app_env,
        settings.port,
        os.getenv("PORT"),
        settings.enable_scheduler,
        STATIC_DIR,
    )

    if settings.enable_scheduler:
        scheduler = build_scheduler()
        scheduler.start()
        logger.info("web_scheduler_started scheduler_timezone=%s", settings.scheduler_timezone)
    else:
        logger.info("web_scheduler_skipped reason=enable_scheduler_false")

    yield

    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("web_scheduler_stopped")


app = FastAPI(title="TW Stock AI", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)
app.include_router(ui_router)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request, exc: Exception):  # type: ignore[no-untyped-def]
    logger.exception("unhandled_exception path=%s error=%s", request.url.path, exc)
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=500, content={"detail": "internal_server_error"})
    return JSONResponse(status_code=500, content={"detail": "internal_server_error"})
