from __future__ import annotations

from math import fabs
from statistics import mean

from tw_stock_ai.config import Settings
from tw_stock_ai.models import PriceBar
from tw_stock_ai.services.short_term_types import IndicatorBundle


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    multiplier = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append((value - result[-1]) * multiplier + result[-1])
    return result


def _rsi(values: list[float], period: int) -> float | None:
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    for gain, loss in zip(gains[period:], losses[period:], strict=False):
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    if len(closes) <= period:
        return None
    true_ranges: list[float] = []
    for index in range(1, len(closes)):
        true_ranges.append(
            max(
                highs[index] - lows[index],
                fabs(highs[index] - closes[index - 1]),
                fabs(lows[index] - closes[index - 1]),
            )
        )
    atr_value = mean(true_ranges[:period])
    for tr in true_ranges[period:]:
        atr_value = ((atr_value * (period - 1)) + tr) / period
    return atr_value


def _adx(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    if len(closes) <= period * 2:
        return None

    tr_list: list[float] = []
    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []
    for index in range(1, len(closes)):
        up_move = highs[index] - highs[index - 1]
        down_move = lows[index - 1] - lows[index]
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
        tr = max(
            highs[index] - lows[index],
            fabs(highs[index] - closes[index - 1]),
            fabs(lows[index] - closes[index - 1]),
        )
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    tr_smooth = sum(tr_list[:period])
    plus_dm_smooth = sum(plus_dm_list[:period])
    minus_dm_smooth = sum(minus_dm_list[:period])
    dx_values: list[float] = []

    for index in range(period, len(tr_list)):
        if index > period:
            tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[index]
            plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm_list[index]
            minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm_list[index]
        plus_di = 100 * (plus_dm_smooth / tr_smooth) if tr_smooth else 0.0
        minus_di = 100 * (minus_dm_smooth / tr_smooth) if tr_smooth else 0.0
        denominator = plus_di + minus_di
        dx = 100 * fabs(plus_di - minus_di) / denominator if denominator else 0.0
        dx_values.append(dx)

    if len(dx_values) < period:
        return None
    adx_value = mean(dx_values[:period])
    for dx in dx_values[period:]:
        adx_value = ((adx_value * (period - 1)) + dx) / period
    return adx_value


def calculate_indicator_bundle(bars: list[PriceBar], settings: Settings) -> IndicatorBundle | None:
    ordered = sorted(bars, key=lambda item: item.trade_date)
    if len(ordered) < settings.min_price_bars_for_screening:
        return None

    closes = [float(item.close) for item in ordered]
    highs = [float(item.high) for item in ordered]
    lows = [float(item.low) for item in ordered]
    volumes = [item.volume for item in ordered]

    ema20 = _ema_series(closes, settings.indicator_ema_fast_period)[-1]
    ema60 = _ema_series(closes, settings.indicator_ema_slow_period)[-1]
    rsi = _rsi(closes, settings.indicator_rsi_period)

    macd_fast = _ema_series(closes, settings.indicator_macd_fast_period)
    macd_slow = _ema_series(closes, settings.indicator_macd_slow_period)
    macd_series = [fast - slow for fast, slow in zip(macd_fast, macd_slow, strict=False)]
    macd_signal_series = _ema_series(macd_series, settings.indicator_macd_signal_period)
    macd_line = macd_series[-1] if macd_series else None
    macd_signal = macd_signal_series[-1] if macd_signal_series else None
    macd_histogram = (
        macd_line - macd_signal if macd_line is not None and macd_signal is not None else None
    )

    adx = _adx(highs, lows, closes, settings.indicator_adx_period)
    atr = _atr(highs, lows, closes, settings.indicator_atr_period)

    recent_volume_slice = volumes[-20:]
    average_volume_20d = mean(recent_volume_slice) if recent_volume_slice else None
    volume_ratio = (
        volumes[-1] / average_volume_20d if average_volume_20d and average_volume_20d > 0 else None
    )
    turnover_values = [close * volume for close, volume in zip(closes[-20:], volumes[-20:], strict=False)]
    average_turnover_20d = mean(turnover_values) if turnover_values else None

    recent_high_20d = max(highs[-settings.breakout_lookback_days :])
    recent_low_20d = min(lows[-settings.breakout_lookback_days :])
    consolidation_high = max(highs[-settings.consolidation_lookback_days :])
    consolidation_low = min(lows[-settings.consolidation_lookback_days :])
    consolidation_range_ratio = (
        (consolidation_high - consolidation_low) / closes[-1] if closes[-1] > 0 else None
    )

    return IndicatorBundle(
        latest_close=closes[-1],
        latest_volume=volumes[-1],
        ema20=ema20,
        ema60=ema60,
        rsi=rsi,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        adx=adx,
        atr=atr,
        average_volume_20d=average_volume_20d,
        average_turnover_20d=average_turnover_20d,
        volume_ratio=volume_ratio,
        recent_high_20d=recent_high_20d,
        recent_low_20d=recent_low_20d,
        consolidation_range_ratio=consolidation_range_ratio,
    )
