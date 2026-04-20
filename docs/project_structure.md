# Project Structure

## Root

```text
.
├─ data/
│  ├─ import_templates/
│  └─ logs/
├─ docs/
├─ scripts/
├─ src/
│  └─ tw_stock_ai/
│     ├─ adapters/
│     ├─ ai_adapters/
│     ├─ notifiers/
│     ├─ prompts/
│     ├─ routers/
│     ├─ services/
│     ├─ static/
│     └─ templates/
├─ tests/
├─ Dockerfile
├─ worker.Dockerfile
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
└─ .env.example
```

## Module Ownership

- `src/tw_stock_ai/config.py`
  - centralized settings loading from `.env`
- `src/tw_stock_ai/db.py`
  - engine, session, database initialization, runtime directories
- `src/tw_stock_ai/models.py`
  - SQLAlchemy schema definitions
- `src/tw_stock_ai/schemas.py`
  - Pydantic API payload models
- `src/tw_stock_ai/adapters/`
  - replaceable market and data-source adapters
- `src/tw_stock_ai/ai_adapters/`
  - replaceable AI provider adapters
- `src/tw_stock_ai/notifiers/`
  - replaceable notification channels
- `src/tw_stock_ai/services/`
  - screening, scoring, holdings, cost control, feature flags, rate limits, logging, jobs
- `src/tw_stock_ai/routers/`
  - REST API and server-rendered HTML pages
- `src/tw_stock_ai/templates/`
  - formal UI pages
- `src/tw_stock_ai/static/`
  - CSS and static assets
- `tests/`
  - regression tests for data, scoring, AI, cost controls, APIs, and UI
