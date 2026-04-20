from __future__ import annotations

import csv
import json
from datetime import date, datetime, time, timezone
from io import StringIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class HttpFetchError(RuntimeError):
    pass


def http_get_bytes(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> bytes:
    request = Request(url, headers=headers or {}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HttpFetchError(f"http_error:{exc.code}:{body[:200]}") from exc
    except URLError as exc:
        raise HttpFetchError(f"url_error:{exc.reason}") from exc


def http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HttpFetchError(f"http_error:{exc.code}:{body[:400]}") from exc
    except URLError as exc:
        raise HttpFetchError(f"url_error:{exc.reason}") from exc
    return json.loads(decode_text(raw))


def http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    return json.loads(decode_text(http_get_bytes(url, headers=headers, timeout=timeout)))


def http_get_csv_rows(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    skip_lines: int = 0,
    encoding: str | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    text = decode_text(http_get_bytes(url, headers=headers, timeout=timeout), preferred_encoding=encoding)
    if skip_lines:
        text = "\n".join(text.splitlines()[skip_lines:])
    reader = csv.DictReader(StringIO(text))
    rows = [dict(row) for row in reader]
    return rows, list(reader.fieldnames or [])


def decode_text(raw: bytes, preferred_encoding: str | None = None) -> str:
    encodings = []
    if preferred_encoding:
        encodings.append(preferred_encoding)
    encodings.extend(["utf-8-sig", "cp950", "utf-8", "big5"])
    seen: set[str] = set()
    for encoding in encodings:
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def build_url(base_url: str, path: str = "", params: dict[str, Any] | None = None) -> str:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" if path else base_url
    if not params:
        return url
    query = urlencode({key: value for key, value in params.items() if value not in (None, "")})
    return f"{url}?{query}"


def quote_path(value: str) -> str:
    return quote(value, safe="")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "")
    if cleaned in {"", "-", "--", "N/A", "nan", "None"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def roc_date_to_date(value: str | None) -> date | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if len(digits) != 7:
        return None
    return date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7]))


def roc_year_month_to_date(value: str | None) -> date | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if len(digits) != 5:
        return None
    return date(int(digits[:3]) + 1911, int(digits[3:5]), 1)


def parse_iso_or_date(value: str | None) -> date | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if "T" in cleaned:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
    return date.fromisoformat(cleaned)


def roc_datetime_to_utc(roc_date_value: str | None, roc_time_value: str | None) -> datetime | None:
    trade_date = roc_date_to_date(roc_date_value)
    if trade_date is None:
        return None
    digits = "".join(ch for ch in str(roc_time_value or "").strip() if ch.isdigit())
    if len(digits) < 6:
        return datetime.combine(trade_date, time(0, 0), tzinfo=timezone.utc)
    return datetime(
        trade_date.year,
        trade_date.month,
        trade_date.day,
        int(digits[:2]),
        int(digits[2:4]),
        int(digits[4:6]),
        tzinfo=timezone.utc,
    )
