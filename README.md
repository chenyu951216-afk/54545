# TW Stock AI

台股短線 3-10 天選股、寶藏股欄位、持股追蹤、Discord 推播、Web UI 與 Zeabur 部署的可執行骨架。

這一版的目標不是硬上所有高級功能，而是先建立一個正式、可跑、可擴展、且不會亂造資料的基礎架構。

## Core Principles

- Never fabricate stock, news, revenue, financial, or indicator results.
- Never assume third-party API response fields.
- Return `missing` or `unavailable` when real data is insufficient.
- Keep every score explainable through sub-scores and evidence.
- Preserve source name, source URL, and fetched time.
- Build modularly so providers and strategy layers can be swapped later.

## Tech Stack

- Python
- FastAPI
- SQLAlchemy
- SQLite first
- PostgreSQL-ready
- APScheduler
- Discord Webhook
- Server-rendered Web UI
- Docker / Zeabur deployment

## What Is Implemented

- Executable FastAPI backend
- SQLite-backed schema with PostgreSQL-compatible ORM design
- PostgreSQL driver included for Zeabur production deployment
- Manual CSV ingestion adapter for verified daily price data
- Real provider adapters for official / documented sources:
  - Fugle historical prices and volumes
  - MOPS listed-company monthly revenue
  - MOPS listed-company daily important information
  - MOPS / TWSE listed-company profiles, EPS, income statement, balance sheet, PE/PB/dividend yield
  - TWSE holiday schedule
- Rule-based short-term screening engine for 3-10 day trading candidates
- Treasure/value scoring engine
- Holdings tracking and exit monitoring
- Discord push service
- Daily Discord report generator with retry and delivery logs
- Separate scheduler worker entrypoint
- Automatic refresh pipeline for Fugle / MOPS / TWSE data before scheduled screening
- Formal Web UI pages for picks, treasures, holdings, settings, and system status
- Dockerfiles for `web` and `worker`
- Environment-driven config
- App-level runtime setting overrides stored in database
- Cost control stack: rate limits, AI whitelist, feature flags, AI cache, usage tracking, cost dashboard
- Structured logging to `data/logs/`
- Replaceable notification abstraction and AI adapter abstraction
- OpenAI Responses API adapter for evidence-based explanation generation

## Project Structure

See:

- [Project Structure](docs/project_structure.md)
- [System Spec](docs/spec.md)
- [Data Flow](docs/data_flow.md)
- [Data Refresh Flow](docs/data_refresh_flow.md)
- [DB Schema](docs/db_schema.md)
- [API Design](docs/api_design.md)
- [Cost Control](docs/cost_control.md)
- [Zeabur Deployment](docs/zeabur.md)

## Local Setup

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run Web

```bash
uvicorn tw_stock_ai.main:app --reload
```

Web URLs:

- UI: `http://127.0.0.1:8000/picks`
- API docs: `http://127.0.0.1:8000/docs`

UI pages:

- `/picks`
- `/treasures`
- `/holdings`
- `/settings`
- `/system`

Operational APIs:

- `/api/system/costs`
- `/api/settings/effective`

## Run Worker

```bash
python -m tw_stock_ai.worker
```

The worker is responsible for cron-like scheduling and Discord pushes.
It now refreshes real market data into the database before screening and reporting.

## Environment Variables

Copy `.env.example` and adjust values.

Key settings:

- `DATABASE_URL`
- `DISCORD_WEBHOOK_URL`
- `DISCORD_ENABLED`
- `DISCORD_TIMEOUT_SECONDS`
- `DISCORD_RETRY_ATTEMPTS`
- `DISCORD_RETRY_BACKOFF_SECONDS`
- `DISCORD_DAILY_REPORT_TOP_N`
- `ENABLE_SCHEDULER`
- `SCHEDULER_TIMEZONE`
- `SCREENING_HOUR`
- `SCREENING_MINUTE`
- `SCREENING_WEEKDAYS`
- `SCREENING_TOP_N`
- `MIN_PRICE_BARS_FOR_SCREENING`
- `AI_ENABLED`
- `NEWS_ANALYSIS_ENABLED`
- `FEATURE_COST_GUARDRAILS_ENABLED`
- `FEATURE_CANDIDATE_AI_ANALYSIS_ENABLED`
- `FEATURE_HOLDING_AI_ANALYSIS_ENABLED`
- `FEATURE_DISCORD_NOTIFICATIONS_ENABLED`
- `OVERALL_MONTHLY_BUDGET_TWD`
- `AI_MONTHLY_BUDGET_TWD`
- `API_RATE_LIMIT_WINDOW_MINUTES`
- `RATE_LIMIT_SCREENING_RUNS_PER_WINDOW`
- `RATE_LIMIT_DISCORD_REPORTS_PER_WINDOW`
- `RATE_LIMIT_CANDIDATE_AI_CALLS_PER_WINDOW`
- `RATE_LIMIT_HOLDING_AI_CALLS_PER_WINDOW`

Secrets guidance:

- keep production secrets in Zeabur environment variables
- do not commit `.env`
- the settings page can store override values, but production secrets should still be managed through env vars whenever possible

## CSV Template

Use:

- [daily_prices.csv](data/import_templates/daily_prices.csv)

Required headers:

- `symbol`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `source_name`
- `source_url`
- `fetched_at`

## Current Data Strategy

This project now supports verified real-data adapters for:

- `fugle` for historical price / volume
- `mops_listed_monthly_revenue` for listed-company revenue
- `mops_listed_daily_info` for listed-company important disclosures
- `twse_mops_listed` for listed-company profile / EPS / income statement / balance sheet / PE / PB / dividend yield
- `twse_holiday_schedule` for official TWSE open/close days
- `manual_csv` for manual verified imports

Still intentionally not hard-coded:

- any commercial paid Taiwan finance news API whose field contract has not been verified in this repo
- any provider response field that has not been validated

When evidence is insufficient, the system returns `missing` / `unavailable` instead of filling gaps.

## Implemented APIs

- `GET /api/health`
- `POST /api/ingestion/manual/prices`
- `POST /api/data-refresh/run`
- `GET /api/data-refresh/latest`
- `GET /api/data-refresh/runs`
- `POST /api/screenings/run`
- `GET /api/screenings/latest`
- `GET /api/screenings/runs`
- `POST /api/discord/reports/run`
- `GET /api/discord/reports/latest`
- `GET /api/system/costs`
- `GET /api/settings/effective`
- `GET /api/holdings`
- `POST /api/holdings`

## Implemented UI Pages

- `GET /picks`
- `GET /treasures`
- `GET /holdings`
- `GET /settings`
- `GET /system`

## Tests

```bash
py -m pytest tests
```

Current regression coverage includes:

- data refresh and cache
- refresh request bootstrap / incremental overlap logic
- short-term scoring
- value engine
- AI analysis
- cost controls and rate limits
- Web UI routing
- system APIs

## Docker

Web:

```bash
docker build -t tw-stock-ai-web .
docker run -p 8000:8000 --env-file .env tw-stock-ai-web
```

Worker:

```bash
docker build -f worker.Dockerfile -t tw-stock-ai-worker .
docker run --env-file .env tw-stock-ai-worker
```

## Zeabur

Recommended services:

1. `web` using root `Dockerfile`
2. `worker` using `worker.Dockerfile`
3. PostgreSQL service for production

Important:

- do not deploy this project through repeated `Add files via upload`
- deploy from the GitHub repository so Zeabur always receives the complete project tree
- `web` and `worker` use the same repository but different Dockerfiles and different `ENABLE_SCHEDULER` values

See full setup:

- [docs/zeabur.md](docs/zeabur.md)
