# Database Schema

## Database Strategy

- Start with SQLite: `sqlite:///./data/app.db`
- Upgrade path: PostgreSQL by changing `DATABASE_URL`
- ORM: SQLAlchemy 2.x
- JSON fields use generic SQLAlchemy `JSON` type for SQLite/PostgreSQL compatibility

## Tables

### `data_sources`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `name` | string(100) | unique provider name |
| `source_type` | string(50) | `manual_csv`, future provider type |
| `base_url` | string(500) nullable | provider homepage or endpoint root |
| `status` | string(30) | `active`, `inactive` |
| `notes` | text nullable | operator notes |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

### `price_bars`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `symbol` | string(20) | stock symbol |
| `trade_date` | date | trading date |
| `open` | numeric(12,4) | real price |
| `high` | numeric(12,4) | real price |
| `low` | numeric(12,4) | real price |
| `close` | numeric(12,4) | real price |
| `volume` | integer | traded volume |
| `source_name` | string(100) | provider name |
| `source_url` | string(500) | evidence URL |
| `fetched_at` | datetime | fetch timestamp |
| `raw_payload` | json | original row payload |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

Constraints:

- unique on `symbol + trade_date + source_name`

### `fundamental_snapshots`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `symbol` | string(20) | stock symbol |
| `snapshot_date` | date | statement or snapshot date |
| `source_name` | string(100) | provider name |
| `source_url` | string(500) | evidence URL |
| `fetched_at` | datetime | fetch timestamp |
| `revenue_yoy` | float nullable | optional |
| `gross_margin` | float nullable | optional |
| `operating_margin` | float nullable | optional |
| `debt_ratio` | float nullable | optional |
| `raw_payload` | json | original payload |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

### `news_items`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `symbol` | string(20) nullable | matched symbol if known |
| `title` | string(300) | raw title |
| `source_name` | string(100) | provider name |
| `source_url` | string(500) | evidence URL |
| `published_at` | datetime | publish time |
| `raw_payload` | json | original payload |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

### `screening_runs`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `as_of_date` | date | latest trade date used |
| `status` | string(30) | `completed`, future `failed` |
| `universe_size` | integer | symbol count |
| `notes` | text nullable | missing data or run notes |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

### `screening_candidates`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `run_id` | integer | screening run id |
| `symbol` | string(20) | stock symbol |
| `status` | string(30) | `ready`, `missing_data` |
| `overall_score` | float nullable | total score |
| `sub_scores` | json | score breakdown |
| `evidence` | json | supporting metrics and timestamps |
| `entry_zone_low` | float nullable | suggested entry lower bound |
| `entry_zone_high` | float nullable | suggested entry upper bound |
| `stop_loss` | float nullable | stop loss |
| `take_profit` | float nullable | take profit |
| `risk_reward_ratio` | float nullable | calculated ratio |
| `treasure_status` | string(30) | `ready`, `unavailable` |
| `treasure_score` | float nullable | optional score |
| `treasure_evidence` | json | fundamentals evidence |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |

### `holdings`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | primary key |
| `symbol` | string(20) | stock symbol |
| `quantity` | integer | position size |
| `average_cost` | float | user-entered average cost |
| `note` | text nullable | notes |
| `created_at` | datetime | created timestamp |
| `updated_at` | datetime | updated timestamp |
