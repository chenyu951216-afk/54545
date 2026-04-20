from typing import Dict, Any, List

from clients.okx_client import OKXClient
from config.settings import settings


class PositionSyncService:
    def __init__(self) -> None:
        self.client = OKXClient()
        self.instrument_cache: dict[str, dict[str, Any]] = {}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, "", "None"):
                return default
            return float(value)
        except Exception:
            return default

    def _load_instrument(self, inst_id: str) -> dict[str, Any]:
        cached = self.instrument_cache.get(inst_id)
        if cached:
            return cached
        for row in self.client.safe_get_instruments():
            if str(row.get("instId", "")).upper() == str(inst_id).upper():
                self.instrument_cache[inst_id] = row
                return row
        return {}

    def _normalize_side(self, row: Dict[str, Any]) -> str:
        pos_side = str(row.get("posSide", "") or "").lower()
        if pos_side in {"long", "short"}:
            return pos_side
        pos = self._safe_float(row.get("pos"), 0.0)
        return "short" if pos < 0 else "long"

    def sync(self) -> List[Dict[str, Any]]:
        if not settings.enable_position_sync:
            return []
        payload = self.client.safe_get_positions()
        out: List[Dict[str, Any]] = []
        for row in payload.get("data", []):
            try:
                symbol = str(row.get("instId", "") or "")
                inst = self._load_instrument(symbol) if symbol else {}
                ct_val = self._safe_float(inst.get("ctVal"), 1.0)
                ct_mult = self._safe_float(inst.get("ctMult"), 1.0)
                contract_unit = ct_val * ct_mult if ct_val > 0 and ct_mult > 0 else 1.0
                size = abs(self._safe_float(row.get("pos"), 0.0))
                entry_price = self._safe_float(row.get("avgPx"), 0.0)
                current_price = self._safe_float(row.get("markPx"), entry_price)
                leverage = self._safe_float(row.get("lever"), 0.0)
                notional = size * contract_unit * current_price
                margin_used = self._safe_float(row.get("margin"), 0.0)
                if margin_used <= 0 and leverage > 0:
                    margin_used = notional / leverage
                out.append({
                    "symbol": symbol,
                    "side": self._normalize_side(row),
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "upl": self._safe_float(row.get("upl"), 0.0),
                    "upl_ratio": self._safe_float(row.get("uplRatio"), 0.0),
                    "leverage": leverage,
                    "margin_used": margin_used,
                    "notional_usdt": notional,
                    "contract_unit": contract_unit,
                    "instId": symbol,
                    "posSide": row.get("posSide", ""),
                    "pos": size,
                    "avgPx": entry_price,
                    "markPx": current_price,
                })
            except (TypeError, ValueError):
                continue
        return out
