from __future__ import annotations

import csv
import io
from datetime import date, datetime
from decimal import Decimal
from typing import BinaryIO

from sqlalchemy import select

from tw_stock_ai.db import SessionLocal
from tw_stock_ai.models import DataSource, PriceBar
from tw_stock_ai.schemas import ImportResult


class ManualCsvPriceAdapter:
    adapter_name = "manual_csv"

    required_headers = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_name",
        "source_url",
        "fetched_at",
    }

    def ingest(self, file_obj: BinaryIO) -> ImportResult:
        content = file_obj.read()
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        headers = set(reader.fieldnames or [])
        missing = sorted(self.required_headers - headers)
        if missing:
            return ImportResult(
                adapter=self.adapter_name,
                records_received=0,
                records_inserted=0,
                records_skipped=0,
                status="failed",
                detail=f"missing headers: {', '.join(missing)}",
            )

        inserted = 0
        skipped = 0
        rows = list(reader)
        with SessionLocal() as session:
            for row in rows:
                symbol = row["symbol"].strip()
                trade_date = date.fromisoformat(row["trade_date"].strip())
                source_name = row["source_name"].strip()

                existing = session.scalar(
                    select(PriceBar).where(
                        PriceBar.symbol == symbol,
                        PriceBar.trade_date == trade_date,
                        PriceBar.source_name == source_name,
                    )
                )
                if existing:
                    skipped += 1
                    continue

                data_source = session.scalar(
                    select(DataSource).where(DataSource.name == source_name)
                )
                if data_source is None:
                    data_source = DataSource(
                        name=source_name,
                        source_type="manual_csv",
                        base_url=row["source_url"].strip(),
                        status="active",
                    )
                    session.add(data_source)

                session.add(
                    PriceBar(
                        symbol=symbol,
                        trade_date=trade_date,
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=int(row["volume"]),
                        source_name=source_name,
                        source_url=row["source_url"].strip(),
                        fetched_at=datetime.fromisoformat(row["fetched_at"].strip()),
                        raw_payload=dict(row),
                    )
                )
                inserted += 1

            session.commit()

        return ImportResult(
            adapter=self.adapter_name,
            records_received=len(rows),
            records_inserted=inserted,
            records_skipped=skipped,
            status="success",
            detail=None,
        )
