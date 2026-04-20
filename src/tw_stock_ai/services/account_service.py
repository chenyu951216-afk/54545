from typing import Dict, Any
import time

from clients.okx_client import OKXClient
from config.settings import settings


class AccountService:
    def __init__(self) -> None:
        self.client = OKXClient()

    def _normalize_pos_mode(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"net", "net_mode", "single", "single_pos_mode"}:
            return "net"
        if raw in {"long_short", "long_short_mode", "hedge", "buy_sell_mode"}:
            return "long_short"
        return "net"

    def credentials_ready(self) -> bool:
        return all([settings.okx_api_key, settings.okx_api_secret, settings.okx_api_passphrase])

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def get_account_summary(self) -> Dict[str, float]:
        payload = self.client.safe_get_balance()
        rows = payload.get("data", [])
        if not rows:
            return {"equity": 0.0, "available": 0.0, "used_margin": 0.0}

        row = rows[0]
        details = row.get("details", []) or []

        usdt_detail = None
        for d in details:
            if str(d.get("ccy", "")).upper() == "USDT":
                usdt_detail = d
                break

        if usdt_detail:
            equity = self._to_float(usdt_detail.get("eq") or row.get("totalEq"))
            available = self._to_float(usdt_detail.get("availBal") or usdt_detail.get("availEq") or row.get("adjEq"))
            used_margin = self._to_float(usdt_detail.get("imr") or usdt_detail.get("frozenBal"))
        else:
            equity = self._to_float(row.get("totalEq"))
            available = self._to_float(row.get("adjEq"))
            used_margin = self._to_float(row.get("imr"))

        if used_margin <= 0:
            used_margin = max(equity - available, 0.0)

        return {
            "equity": equity,
            "available": available,
            "used_margin": used_margin,
        }

    def summary(self) -> Dict[str, Any]:
        account = self.get_account_summary()
        config = self.client.safe_get_account_config() if self.credentials_ready() else {"code": "-1", "data": []}
        rows = config.get("data", []) if isinstance(config, dict) else []
        config_row = rows[0] if rows and isinstance(rows[0], dict) else {}
        pos_mode = self._normalize_pos_mode(config_row.get("posMode"))
        return {
            "equity": float(account.get("equity", 0.0)),
            "available": float(account.get("available", 0.0)),
            "available_equity": float(account.get("available", 0.0)),
            "used_margin": float(account.get("used_margin", 0.0)),
            "timestamp": int(time.time()),
            "credentials_ready": self.credentials_ready(),
            "pos_mode": pos_mode,
            "account_pos_mode_raw": str(config_row.get("posMode", "") or ""),
        }
