import math
from typing import List, Tuple


def ema(prices: List[float], period: int) -> List[float]:
    n = len(prices)
    if n == 0 or period < 1:
        return []
    result = [float('nan')] * n
    if period > n:
        return result
    alpha = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    seed = sum(prices[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, n):
        val = alpha * prices[i] + (1.0 - alpha) * prev
        result[i] = val
        prev = val
    return result


def sma(prices: List[float], period: int) -> List[float]:
    n = len(prices)
    if n == 0 or period < 1:
        return []
    result = [float('nan')] * n
    if period > n:
        return result
    window_sum = sum(prices[:period])
    result[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += prices[i] - prices[i - period]
        result[i] = window_sum / period
    return result


def rsi(prices: List[float], period: int = 14) -> List[float]:
    n = len(prices)
    if n == 0 or period < 1:
        return []
    result = [float('nan')] * n
    if n < period + 1:
        return result
    # Calculate price changes
    gains = []
    losses = []
    for i in range(1, n):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    # First average: simple mean over initial `period` changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    # Wilder's smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    n = len(prices)
    if n == 0 or fast < 1 or slow < 1 or signal_period < 1:
        return [], [], []
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = [float('nan')] * n
    for i in range(n):
        if math.isnan(fast_ema[i]) or math.isnan(slow_ema[i]):
            continue
        macd_line[i] = fast_ema[i] - slow_ema[i]
    # Collect non-NaN MACD values for signal EMA
    macd_valid_start = -1
    for i in range(n):
        if not math.isnan(macd_line[i]):
            macd_valid_start = i
            break
    signal_line = [float('nan')] * n
    histogram = [float('nan')] * n
    if macd_valid_start < 0:
        return macd_line, signal_line, histogram
    # Extract valid MACD values and compute EMA of those
    valid_macd = [macd_line[i] for i in range(macd_valid_start, n)]
    valid_signal = ema(valid_macd, signal_period)
    for j, i in enumerate(range(macd_valid_start, n)):
        signal_line[i] = valid_signal[j]
        if not math.isnan(valid_signal[j]):
            histogram[i] = macd_line[i] - valid_signal[j]
    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: List[float],
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    n = len(prices)
    if n == 0 or period < 1:
        return [], [], []
    upper = [float('nan')] * n
    middle = [float('nan')] * n
    lower = [float('nan')] * n
    if period > n:
        return upper, middle, lower
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, middle, lower


def atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[float]:
    n = len(highs)
    if n == 0 or len(lows) != n or len(closes) != n or period < 1:
        return [float('nan')] * max(len(highs), 0)
    result = [float('nan')] * n
    if n < period + 1:
        return result
    # True range: first element has no previous close
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))
    # Seed ATR with simple mean of first `period` true ranges (starting at index 1)
    first_avg = sum(tr[1:period + 1]) / period
    result[period] = first_avg
    prev = first_avg
    # Wilder's smoothing
    for i in range(period + 1, n):
        val = (prev * (period - 1) + tr[i]) / period
        result[i] = val
        prev = val
    return result


def vwap(prices: List[float], volumes: List[float]) -> List[float]:
    n = len(prices)
    if n == 0 or len(volumes) != n:
        return [float('nan')] * max(len(prices), 0)
    result = [float('nan')] * n
    cum_pv = 0.0
    cum_vol = 0.0
    for i in range(n):
        cum_pv += prices[i] * volumes[i]
        cum_vol += volumes[i]
        if cum_vol > 0.0:
            result[i] = cum_pv / cum_vol
    return result
