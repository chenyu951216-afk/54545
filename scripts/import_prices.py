from __future__ import annotations

import argparse
from pathlib import Path

from tw_stock_ai.adapters.manual_csv import ManualCsvPriceAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Import price bars from CSV")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    adapter = ManualCsvPriceAdapter()
    with args.csv_path.open("rb") as handle:
        result = adapter.ingest(handle)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
