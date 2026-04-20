# Change Log

## 2026-04-20

- Fixed short-term risk/reward calculation so it now varies by technical range and risk distance instead of always echoing the configured `1.80`.
- Changed manual `/picks/run` and `/discord/run` UI actions to run in the background so the page no longer hangs while refresh, screening, AI analysis, and Discord dispatch are running.
- Changed Discord delivery retry behavior so fatal 4xx responses such as `403` stop immediately instead of wasting extra retries and making the UI feel stuck.
- Added a premium news window helper for `Asia/Taipei` 07:00-09:00 so free news can run continuously and symbol-scoped premium news can be layered in during the morning window.
- Added recent-news visibility to the picks UI and a new `/api/news/latest` endpoint so live news ingestion can be checked directly.
- Fixed the daily report fallback reason builder so it reads the correct sub-score keys.
