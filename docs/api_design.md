# REST API Design

## Health

### `GET /api/health`

Returns:

- app name
- environment
- database URL
- scheduler enabled flag
- current time

## Ingestion

### `POST /api/ingestion/manual/prices`

Multipart upload:

- `file`: CSV file matching `data/import_templates/daily_prices.csv`

Returns:

- adapter name
- received count
- inserted count
- skipped count
- status
- detail

## Screening

### `POST /api/screenings/run`

Triggers a market-data refresh and then a screening run immediately.

Returns:

- run metadata
- all candidates in the run

### `GET /api/screenings/latest`

Returns latest screening run or `null`.

### `GET /api/screenings/runs`

Returns up to 20 historical runs.

Rate limit:

- high-cost manual screening calls are rate-limited

## Data Refresh

### `POST /api/data-refresh/run`

Triggers a forced market-data refresh immediately.

Returns:

- refresh run metadata
- per-dataset refresh item status
- received / cleaned / stored counts

### `GET /api/data-refresh/latest`

Returns latest refresh run or `null`.

### `GET /api/data-refresh/runs`

Returns up to 20 historical refresh runs.

## Discord Reports

### `POST /api/discord/reports/run`

Triggers a fresh screening run, generates the daily Discord report, and sends it through the configured webhook.

Returns:

- report date
- screening run id
- qualified count
- rendered Discord content
- delivery logs with retry history

### `GET /api/discord/reports/latest`

Returns the latest Discord report log or `null`.

## System

### `GET /api/system/costs`

Returns:

- monthly budget
- monthly estimated cost
- AI actual cost
- daily and monthly usage counts
- feature flag status

### `GET /api/settings/effective`

Returns:

- effective runtime settings grouped for UI
- schedule preview
- weight sum

## Holdings

### `POST /api/holdings`

JSON body:

```json
{
  "symbol": "2330",
  "quantity": 1000,
  "average_cost": 850.5,
  "note": "manual entry"
}
```

### `GET /api/holdings`

Returns holdings enriched with:

- latest close
- unrealized pnl
- trend status
- exit signal
- evidence

### `POST /api/holdings/monitor`

Refreshes only currently held symbols and then recomputes holding monitoring results.

### `POST /api/holdings/{holding_id}/monitor`

Refreshes only the target holding symbol and then recomputes monitoring for that holding.

## Planned Next APIs

- `GET /api/data-sources`
- `GET /api/prices/{symbol}`
- `POST /api/fundamentals/import`
- `POST /api/news/import`
