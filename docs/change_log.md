# Change Log

## 2026-04-20

- Fixed short-term risk/reward calculation so it now varies by technical range and risk distance instead of always echoing the configured `1.80`.
- Changed manual `/picks/run` and `/discord/run` UI actions to run in the background so the page no longer hangs while refresh, screening, AI analysis, and Discord dispatch are running.
- Changed Discord delivery retry behavior so fatal 4xx responses such as `403` stop immediately instead of wasting extra retries and making the UI feel stuck.
- Added a premium news window helper for `Asia/Taipei` 07:00-09:00 so free news can run continuously and symbol-scoped premium news can be layered in during the morning window.
- Added recent-news visibility to the picks UI and a new `/api/news/latest` endpoint so live news ingestion can be checked directly.
- Fixed the daily report fallback reason builder so it reads the correct sub-score keys.

## 2026-04-21

- Fixed `daily_report_runs.trigger_source` overflow by compacting long daily-report trigger source labels such as `ui_manual_dispatch:fallback_prepare` before writing them into the `varchar(30)` column.
- Added regression coverage to keep daily-report trigger sources under the database column limit and to verify the manual Discord dispatch path stores the compacted value.
- Changed the hybrid market-news adapter so free FinMind `TaiwanStockNews` can be polled all day for symbol-scoped scans instead of being limited to the 07:00-09:00 premium window.
- Added a dedicated worker news-polling schedule and targeted news-only refresh path so the system can keep scanning candidate and holding symbols 24/7 without re-running the full price, volume, revenue, and fundamentals refresh stack every time.
- Added Discord delivery diagnostics to the picks and system pages so the latest webhook attempt status, HTTP code, masked webhook target, and response body can be inspected directly from the UI.
- Fixed manual `/discord/run` so UI-triggered pushes bypass the internal `discord_report_run` rate limiter; scheduled pushes still keep their guardrail, but the manual button is no longer skipped just because the hourly counter was full.
