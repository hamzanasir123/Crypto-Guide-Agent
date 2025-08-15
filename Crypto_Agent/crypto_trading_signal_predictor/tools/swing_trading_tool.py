from typing import List, Tuple, Dict, Any, Optional
import datetime
from literalai import TypedDict
import pandas as pd
import numpy as np
import random
import time
from agents import function_tool
from pydantic import BaseModel

# --- Config / Helpers ---
DEFAULT_TICK = 0.5  # BTC/USD default tick size (configurable)
MIN_LOOKBACK = 100
MAX_LOOKBACK = 1000

class OHLCVRow(BaseModel):
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

def round_to_tick(price: float, tick: float = DEFAULT_TICK) -> float:
    return round(round(price / tick) * tick, 8)

def iso_ts_now_utc() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# --- Indicators ---
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr_from_ohlcv(df: pd.DataFrame, length: int = 14) -> float:
    # df must have columns: 'high','low','close'
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(length, min_periods=1).mean().iloc[-1]
    return float(atr)

# --- Swing fractal detection (5-bar fractal) ---
def find_swing_fractals(df: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Returns lists of (index, price) for swing_highs and swing_lows using 5-bar fractal:
    Swing High: H[i] > H[i-1], H[i] > H[i-2], H[i] > H[i+1], H[i] > H[i+2]
    Swing Low:  L[i] < L[i-1], L[i] < L[i-2], L[i] < L[i+1], L[i] < L[i+2]
    Indexes refer to df.index (integer positions).
    """
    highs = df['high'].values
    lows = df['low'].values
    swing_highs = []
    swing_lows = []
    n = len(df)
    for i in range(2, n - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, float(highs[i])))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows

# --- Pivot Points (Classic) using the last completed session (last row of provided OHLCV) ---
def classic_pivots_from_bar(bar: Dict[str, float]) -> Dict[str, float]:
    H = bar['high']; L = bar['low']; C = bar['close']
    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2 * (P - L)
    S3 = L - 2 * (H - P)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2, "R3": R3, "S3": S3}

# --- Fibonacci retracement levels between a and b (a->b impulse) ---
def fib_levels(a: float, b: float) -> Dict[str, float]:
    # Retracement levels from a (start) to b (end). If a < b it's an up leg.
    dif = b - a
    return {
        "0.0": b,
        "23.6": b - dif * 0.236,
        "38.2": b - dif * 0.382,
        "50.0": b - dif * 0.5,
        "61.8": b - dif * 0.618,
        "78.6": b - dif * 0.786,
        "100.0": a
    }

# --- Confluence scoring ---
def compute_confidence(level_price: float, now_price: float, atr: float, confluences: List[str], touches: int = 0) -> int:
    score = 0
    for c in confluences:
        if c == "swing":
            score += 25
        elif c == "pivot":
            score += 20
        elif c == "fib":
            score += 15
        elif c == "ema":
            score += 15
        elif c == "proximity":
            score += 15
        elif c == "touch":
            score += 10
    # also add touch counts up to 10 each (already partly in above)
    score += min(10, touches * 2)
    if score > 100:
        score = 100
    return int(score)

# --- Main function implementing contract ---
@function_tool
def swing_trading_tool(
    symbol: str,
    now_price: float,
    timeframe: str = "1h",
    lookback_bars: int = 200,
    ohlcv: List[OHLCVRow] = None,
    tick_size: float = DEFAULT_TICK
) -> Dict[str, Any]:
    """
    Swing trading level generator — tool wrapper for agents.framework.

    Args:
        symbol: market symbol, e.g., "BTC/USD"
        now_price: current last price (float)
        timeframe: one of '15m','1h','4h','1d'
        lookback_bars: number of bars to use from provided OHLCV (will be clamped to [MIN_LOOKBACK,MAX_LOOKBACK])
        ohlcv: list of rows [timestamp, open, high, low, close, volume]
        tick_size: tick size for rounding

    Returns:
        JSON-serializable dict with levels, zones, risk, and a brief human_summary.
    """
    # --- Input validation ---
    if ohlcv is None:
        return {"status": "error", "message": "Missing required input: ohlcv (list of candles)."}
    if symbol is None or not isinstance(symbol, str) or symbol.strip() == "":
        return {"status": "error", "message": "Missing or invalid 'symbol' input."}
    if now_price is None or not isinstance(now_price, (int, float)):
        return {"status": "error", "message": "Missing or invalid 'now_price' input."}
    if timeframe not in ["15m", "1h", "4h", "1d"]:
        return {"status": "error", "message": f"timeframe must be one of ['15m','1h','4h','1d']. Got '{timeframe}'."}
    if lookback_bars is None:
        lookback_bars = 200
    lookback_bars = max(MIN_LOOKBACK, min(MAX_LOOKBACK, int(lookback_bars)))

    # Prepare DataFrame
    # ohlcv rows: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv[-lookback_bars:], columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    # Ensure numeric types
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna().reset_index(drop=True)
    if len(df) < 30:
        return {"status": "error", "message": f"Not enough OHLCV rows after cleaning. Need >=30, got {len(df)}."}

    # Last completed bar for pivots = last row
    last_bar = {"high": float(df['high'].iloc[-1]), "low": float(df['low'].iloc[-1]), "close": float(df['close'].iloc[-1])}
    pivots = classic_pivots_from_bar(last_bar)

    # ATR(14)
    atr = atr_from_ohlcv(df, length=14)
    if atr <= 0:
        return {"status": "error", "message": "Computed ATR is non-positive; check OHLCV input."}
    # EMAs
    ema20_series = ema(df['close'], 20)
    ema50_series = ema(df['close'], 50)
    ema200_series = ema(df['close'], 200)
    ema20 = float(ema20_series.iloc[-1])
    ema50 = float(ema50_series.iloc[-1])
    ema200 = float(ema200_series.iloc[-1])

    # Trend state
    if ema20 > ema50 > ema200:
        trend_state = "bullish"
    elif ema20 < ema50 < ema200:
        trend_state = "bearish"
    else:
        trend_state = "mixed"

    # Swing fractals
    swing_highs, swing_lows = find_swing_fractals(df)
    # Keep the most recent two swings of each type (if available)
    recent_swing_highs = swing_highs[-2:] if len(swing_highs) >= 2 else swing_highs[:]
    recent_swing_lows = swing_lows[-2:] if len(swing_lows) >= 2 else swing_lows[:]

    # Determine last impulse leg:
    # if price above EMA20 and last swing was Low -> High, use that low->high leg else high->low
    # We'll attempt to find the last swing low and subsequent swing high (their indices)
    impulse = None
    direction = None
    # Build arrays for convenience
    sh_idx = [i for i, p in swing_highs]
    sl_idx = [i for i, p in swing_lows]

    # find last swing order by index:
    last_swing_idx = None
    last_swing_type = None
    if len(sh_idx) or len(sl_idx):
        # pick most recent swing overall
        last_sh = sh_idx[-1] if len(sh_idx) else -1
        last_sl = sl_idx[-1] if len(sl_idx) else -1
        if last_sh > last_sl:
            last_swing_idx = last_sh; last_swing_type = "high"
        else:
            last_swing_idx = last_sl; last_swing_type = "low"

    # find preceding opposite swing to form a leg
    if last_swing_type == "low":
        # find next swing high after that low
        candidate_highs = [(i,p) for (i,p) in swing_highs if i > last_swing_idx]
        if candidate_highs:
            sh_i, sh_p = candidate_highs[0]
            sl_i, sl_p = ([x for x in swing_lows if x[0] == last_swing_idx] or [(last_swing_idx, float(df['low'].iloc[last_swing_idx]))])[0]
            impulse = (sl_p, sh_p); direction = "up"
    elif last_swing_type == "high":
        candidate_lows = [(i,p) for (i,p) in swing_lows if i > last_swing_idx]
        if candidate_lows:
            sl_i, sl_p = candidate_lows[0]
            sh_i, sh_p = ([x for x in swing_highs if x[0] == last_swing_idx] or [(last_swing_idx, float(df['high'].iloc[last_swing_idx]))])[0]
            impulse = (sh_p, sl_p); direction = "down"

    # Fallback: if no clear fractal leg, pick most recent significant swing pair or whole lookback range
    if impulse is None:
        # use last local min and max in the window
        window_low = float(df['low'].min())
        window_high = float(df['high'].max())
        if now_price >= ema20:
            impulse = (window_low, window_high); direction = "up"
        else:
            impulse = (window_high, window_low); direction = "down"

    a_price, b_price = impulse  # start->end of impulse
    fibs = fib_levels(a_price, b_price)

    # Candidate supports and resistances
    candidates_supports = []
    candidates_resistances = []

    # include recent swing lows/highs
    for idx, price in recent_swing_lows:
        candidates_supports.append(("Swing Low", price, ["swing"]))
    for idx, price in recent_swing_highs:
        candidates_resistances.append(("Swing High", price, ["swing"]))

    # include pivots S1,S2 and R1,R2
    candidates_supports.append(("S1 (pivot)", pivots["S1"], ["pivot"]))
    candidates_supports.append(("S2 (pivot)", pivots["S2"], ["pivot"]))
    candidates_resistances.append(("R1 (pivot)", pivots["R1"], ["pivot"]))
    candidates_resistances.append(("R2 (pivot)", pivots["R2"], ["pivot"]))

    # include fib retracements relative to current price (below -> supports, above -> resistances)
    for lab, p in fibs.items():
        p_float = float(p)
        if p_float < now_price:
            candidates_supports.append((f"Fib {lab}%", p_float, ["fib"]))
        elif p_float > now_price:
            candidates_resistances.append((f"Fib {lab}%", p_float, ["fib"]))

    # include EMA200 as support/resistance if relevant
    if ema200 < now_price:
        candidates_supports.append(("EMA200", ema200, ["ema"]))
    elif ema200 > now_price:
        candidates_resistances.append(("EMA200", ema200, ["ema"]))

    # deduplicate close prices (within small tolerance)
    def dedupe_candidates(cands: List[Tuple[str, float, List[str]]]) -> List[Dict[str, Any]]:
        result = []
        seen = []
        tol = atr * 0.02  # 2% of ATR tolerance for merging
        for label, price, reasons in cands:
            merged = False
            for s in result:
                if abs(s['price'] - price) <= tol:
                    # merge reasons and possibly keep label with more reasons
                    s['reasons'] = sorted(set(s['reasons'] + reasons))
                    merged = True
                    break
            if not merged:
                result.append({"label": label, "price": float(price), "reasons": reasons.copy()})
        # sort by proximity to now_price ascending
        result.sort(key=lambda x: abs(x['price'] - now_price))
        return result

    supports = dedupe_candidates(candidates_supports)
    resistances = dedupe_candidates(candidates_resistances)

    # Candidate touches: count how many times price touched +/- small band historically
    def count_touches(df: pd.DataFrame, level_price: float, bandwidth: float) -> int:
        low = level_price - bandwidth
        high = level_price + bandwidth
        # a "touch" if candle low <= level <= candle high
        touched = ((df['low'] <= level_price) & (df['high'] >= level_price)).sum()
        return int(touched)

    # Snap buy/sell dynamic zones
    buy_min = now_price - 1.0 * atr
    buy_max = now_price - 0.3 * atr
    sell_min = now_price + 0.3 * atr
    sell_max = now_price + 1.0 * atr

    # If mixed and zones conflict widen by +0.25*ATR each side
    if trend_state == "mixed":
        expand = 0.25 * atr
        buy_min -= expand; buy_max += expand
        sell_min -= expand; sell_max += expand

    # Snap function: if candidate level within 0.25*ATR of boundary, snap boundary to that level
    def snap_boundary(boundary_price: float, candidate_list: List[Dict[str, Any]], side: str) -> float:
        snap_tol = 0.25 * atr
        best = boundary_price
        for cand in candidate_list:
            if abs(cand['price'] - boundary_price) <= snap_tol:
                # pick candidate that's closer
                if abs(cand['price'] - boundary_price) < abs(best - boundary_price):
                    best = cand['price']
        return best

    buy_min_snapped = snap_boundary(buy_min, supports + resistances, "buy_min")
    buy_max_snapped = snap_boundary(buy_max, supports + resistances, "buy_max")
    sell_min_snapped = snap_boundary(sell_min, supports + resistances, "sell_min")
    sell_max_snapped = snap_boundary(sell_max, supports + resistances, "sell_max")

    # Ensure ordering
    buy_zone_min = min(buy_min_snapped, buy_max_snapped)
    buy_zone_max = max(buy_min_snapped, buy_max_snapped)
    sell_zone_min = min(sell_min_snapped, sell_max_snapped)
    sell_zone_max = max(sell_min_snapped, sell_max_snapped)

    # Snap to tick
    buy_zone_min = round_to_tick(buy_zone_min, tick_size)
    buy_zone_max = round_to_tick(buy_zone_max, tick_size)
    sell_zone_min = round_to_tick(sell_zone_min, tick_size)
    sell_zone_max = round_to_tick(sell_zone_max, tick_size)

    # Build final levels with confidence & reasons
    def build_level_entries(cand_list: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
        out = []
        for cand in cand_list:
            price = round_to_tick(cand['price'], tick_size)
            reasons = cand.get('reasons', []).copy()
            # proximity to ATR band?
            proximity = False
            if side == "support":
                if buy_zone_min <= cand['price'] <= buy_zone_max or abs(cand['price'] - buy_zone_min) <= 0.25*atr or abs(cand['price'] - buy_zone_max) <= 0.25*atr:
                    reasons.append("proximity")
                    proximity = True
            else:
                if sell_zone_min <= cand['price'] <= sell_zone_max or abs(cand['price'] - sell_zone_min) <= 0.25*atr or abs(cand['price'] - sell_zone_max) <= 0.25*atr:
                    reasons.append("proximity")
                    proximity = True
            # ema confluence
            if abs(cand['price'] - ema20) <= 0.25*atr:
                reasons.append("ema20")
            if abs(cand['price'] - ema50) <= 0.25*atr:
                reasons.append("ema50")
            if abs(cand['price'] - ema200) <= 0.25*atr:
                reasons.append("ema200")

            # touches:
            touches = count_touches(df, cand['price'], bandwidth=0.02*atr)
            if touches >= 2:
                reasons.append("multi-touch")

            conf = compute_confidence(price, now_price, atr, reasons, touches=touches)
            out.append({"label": cand['label'], "price": float(price), "confidence": conf, "reasons": sorted(list(set(reasons)))})
        # sort by confidence desc then proximity to now
        out.sort(key=lambda x: (-x['confidence'], abs(x['price'] - now_price)))
        return out

    final_supports = build_level_entries(supports, "support")
    final_resistances = build_level_entries(resistances, "resistance")

    # Risk markers
    # For long setups (prefer long if trend bullish), invalidation = nearest strong support below Buy Zone minus 0.25*ATR
    def find_nearest_strong(levels: List[Dict[str,Any]], reference_price: float, direction: str) -> Optional[Dict[str,Any]]:
        # direction: 'below' or 'above'
        candidates = []
        for lvl in levels:
            if direction == 'below' and lvl['price'] < reference_price:
                candidates.append(lvl)
            if direction == 'above' and lvl['price'] > reference_price:
                candidates.append(lvl)
        if not candidates:
            return None
        # choose by highest confidence then proximity
        candidates.sort(key=lambda x: (-x['confidence'], abs(x['price'] - reference_price)))
        return candidates[0]

    if trend_state == "bullish":
        # longs
        invalid_support = find_nearest_strong(final_supports, buy_zone_min, 'below') or (final_supports[0] if final_supports else None)
        if invalid_support:
            invalidation = invalid_support['price'] - 0.25 * atr
        else:
            invalidation = buy_zone_min - 0.25 * atr
    elif trend_state == "bearish":
        invalid_res = find_nearest_strong(final_resistances, sell_zone_max, 'above') or (final_resistances[0] if final_resistances else None)
        if invalid_res:
            invalidation = invalid_res['price'] + 0.25 * atr
        else:
            invalidation = sell_zone_max + 0.25 * atr
    else:
        # mixed: choose a neutral invalidation around buy zone lower
        invalidation = buy_zone_min - 0.25 * atr

    # suggested stop equals invalidation
    suggested_stop = round_to_tick(invalidation, tick_size)

    # TP1 conservative: nearest resistance/support ahead (~0.5*ATR to 1.0*ATR)
    if trend_state == "bullish":
        # target resistances above now_price
        tp1 = now_price + 0.75 * atr
        tp2 = now_price + 1.5 * atr
    elif trend_state == "bearish":
        tp1 = now_price - 0.75 * atr
        tp2 = now_price - 1.5 * atr
    else:
        # mixed — wider
        tp1 = now_price + 0.75 * atr
        tp2 = now_price + 1.5 * atr

    tp1 = round_to_tick(tp1, tick_size)
    tp2 = round_to_tick(tp2, tick_size)

    # Build JSON result
    result = {
        "status": "ok",
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": iso_ts_now_utc(),
        "now_price": float(round_to_tick(now_price, tick_size)),
        "trend": trend_state,
        "atr": float(round(atr, 8)),
        "ema": {"ema20": float(round(ema20, 8)), "ema50": float(round(ema50, 8)), "ema200": float(round(ema200, 8))},
        "levels": {
            "supports": final_supports,
            "resistances": final_resistances
        },
        "zones": {
            "buy": {"min": float(buy_zone_min), "max": float(buy_zone_max), "basis": "ATR pullback + confluence snap"},
            "sell": {"min": float(sell_zone_min), "max": float(sell_zone_max), "basis": "ATR extension + confluence snap"}
        },
        "risk": {
            "invalidation": float(round_to_tick(invalidation, tick_size)),
            "stop": float(round_to_tick(suggested_stop, tick_size)),
            "tp1": float(tp1),
            "tp2": float(tp2)
        },
        "notes": ["Level confidences are confluence-weighted; see reasons array."],
        "disclaimer": "This is educational information, not financial advice."
    }

    # Human summary (markdown) under ~15 lines
    # levels table: show top 2 supports and resistances
    hs_lines = []
    hs_lines.append(f"{symbol} — {timeframe} Swing Levels")
    hs_lines.append(f"Now: {result['now_price']}")
    hs_lines.append(f"Trend: {trend_state.capitalize()} (EMA20={result['ema']['ema20']}, EMA50={result['ema']['ema50']}, EMA200={result['ema']['ema200']})")
    hs_lines.append(f"ATR(14): ~{round(result['atr'], 2)}")
    # Supports
    hs_lines.append("Supports")
    for s in result['levels']['supports'][:2]:
        reasons_short = ",".join(s['reasons'][:3])
        hs_lines.append(f"{s['label']} — {s['price']} (conf {s['confidence']}): {reasons_short}")
    # Resistances
    hs_lines.append("Resistances")
    for r in result['levels']['resistances'][:2]:
        reasons_short = ",".join(r['reasons'][:3])
        hs_lines.append(f"{r['label']} — {r['price']} (conf {r['confidence']}): {reasons_short}")
    hs_lines.append(f"Buy Zone: {result['zones']['buy']['min']} → {result['zones']['buy']['max']} ({result['zones']['buy']['basis']})")
    hs_lines.append(f"Sell Zone: {result['zones']['sell']['min']} → {result['zones']['sell']['max']} ({result['zones']['sell']['basis']})")
    hs_lines.append(f"Invalidation / Stop: {result['risk']['stop']}")
    hs_lines.append(f"TP1 / TP2: {result['risk']['tp1']} / {result['risk']['tp2']}")
    hs_lines.append("Not financial advice.")
    human_summary = "\n".join(hs_lines[:15])  # cap lines to ~15

    result["human_summary"] = human_summary

    return result
