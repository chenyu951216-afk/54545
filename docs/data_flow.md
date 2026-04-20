# Data Flow Design

## Overview

```text
Verified Data Source
    -> Adapter
    -> Validation / normalization
    -> Raw payload persistence
    -> Core tables
    -> Screening service
    -> Candidate table
    -> REST API / Web UI / Discord push
```

## Ingestion Flow

1. User uploads verified CSV or future provider adapter fetches real data.
2. Adapter validates required fields before mapping into internal schema.
3. Original row or payload is stored in `raw_payload`.
4. Normalized records are written into `price_bars` or future snapshot tables.
5. Duplicate rows are skipped based on business keys.

## Screening Flow

1. `screening` job loads latest available price bars.
2. Data is grouped by symbol.
3. Strategy checks whether minimum bar count is satisfied.
4. If not satisfied:
   - candidate status becomes `missing_data`
   - no fake score or fake trade setup is generated
5. If satisfied:
   - sub-scores are calculated
   - entry zone, stop loss, take profit, risk-reward are calculated
   - evidence includes source and timestamps
6. Results are saved into `screening_runs` and `screening_candidates`.

## Holdings Flow

1. User manually adds holdings via API or UI.
2. System reads latest available price bars for the symbol.
3. Trend and exit status are calculated only when data is sufficient.
4. If data is insufficient, status stays `unavailable`.

## Discord Flow

1. Worker scheduler triggers on weekdays at 08:00 Asia/Taipei.
2. Screening run is executed.
3. Top N `ready` candidates are loaded.
4. If no valid candidates exist, system sends a missing-data message instead of a fake list.

## Extension Points

- Add official exchange / broker / data vendor adapters
- Add fundamentals and news snapshot ingestion
- Add conflict detector for multiple providers
- Add AI summary after evidence is already complete
