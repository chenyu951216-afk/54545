# Cost Control

## Goal

Keep the monthly operating cost close to `TWD 1000` by default.

## Guardrail Layers

1. Rule-first pipeline
   - short-term screening runs on rules first
   - AI is only used after candidates are narrowed down
2. AI whitelist
   - only approved prompt names are allowed
   - holdings AI only runs for actual positions
3. Candidate limits
   - AI only analyzes top `N` candidates
   - optional symbol allowlist is supported
4. Rate limits
   - manual screening
   - manual Discord reports
   - candidate AI
   - holding AI
5. Cache
   - adapter cache for data refresh
   - AI analysis cache by target + analysis kind + evidence hash
6. Feature flags
   - high-cost modules can be disabled independently
7. Cost snapshot
   - monthly budget
   - current AI cost
   - daily and monthly usage counts

## Current Tracked Cost Sources

- AI analysis actual cost from `ai_analysis_records`
- external API call estimated cost from `usage_events`
- notification estimated cost from `usage_events`

## Production Recommendations

- keep secrets in Zeabur environment variables
- use PostgreSQL instead of SQLite for shared `web` + `worker`
- review `/system` and `/api/system/costs` regularly
- keep `AI_ENABLED=false` until a real provider budget is validated
