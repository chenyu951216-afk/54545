# Data Refresh Flow

## Purpose

This flow keeps data-source wiring out of business logic.
Screening, holdings, and UI only read normalized tables.
Adapters and refresh orchestration are responsible for source availability, cache, raw persistence, and cleaned persistence.

## Flow

```text
Refresh request
    -> AdapterRegistry selects adapter per dataset
    -> Cache lookup by adapter + request key
    -> Adapter fetch executes only when cache miss or force refresh
    -> Standardized AdapterFetchResult returned
    -> RawDataRecord persistence
    -> Cleaned table persistence
    -> DataRefreshRun / DataRefreshItem logs
    -> Downstream services read normalized tables only
```

## Datasets

- `price`
- `volume`
- `news`
- `revenue`
- `fundamentals`
- `market_calendar`

## Unified Result Contract

Every adapter returns:

- `adapter_name`
- `dataset`
- `status`
- `fetched_at`
- `raw_items`
- `cleaned_items`
- `detail`
- `from_cache`
- `unavailable_reason`
- `errors`
- `metadata`

## Status Semantics

- `ready`
  - source responded and normalized data is available
- `unavailable`
  - source not configured or temporarily unavailable
- `failed`
  - adapter execution raised an error or parsing failed

## Cache Semantics

- cache key is generated from request scope
- cache entry is stored per adapter
- caller can bypass cache using `force_refresh=true`

## Extension Strategy

When connecting a real provider:

1. implement one concrete adapter class
2. keep provider-specific fields inside the adapter
3. map output to normalized cleaned records
4. register the adapter in `AdapterRegistry`
5. keep business logic unchanged
