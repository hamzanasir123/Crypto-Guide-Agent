from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from agents import function_tool
from pydantic import BaseModel

# ===========================
# Helper Functions
# ===========================
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs.iloc[-1]))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return {"macd": macd.iloc[-1], "signal": macd_signal.iloc[-1]}

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean().iloc[-1]


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    class Config:
        extra = 'forbid'

# ===========================
# Swing Trading Signal Tool
# ===========================
@function_tool
def swing_trading_tool(
    pair: str,
    ohlcv: List[Candle],
    risk_reward_target: float = 2.0
) -> Dict[str, Any]:
    """
    Generate actionable swing trading signals based on multiple confluences of technical indicators.
    """
    print(f"Generating swing trading signal for: {pair}")
    df = pd.DataFrame(ohlcv)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    close_prices = df["close"]
    last_price = close_prices.iloc[-1]

    # --- Indicators ---
    rsi = calculate_rsi(close_prices)
    macd_data = calculate_macd(close_prices)
    atr = calculate_atr(df)

    ema_fast = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = close_prices.ewm(span=50, adjust=False).mean().iloc[-1]

    # --- Signal Logic ---
    indicators_used = []
    signal = None

    if rsi < 30 and macd_data["macd"] > macd_data["signal"] and ema_fast > ema_slow:
        signal = "Buy"
        indicators_used.extend(["RSI", "MACD", "EMA Cross"])
    elif rsi > 70 and macd_data["macd"] < macd_data["signal"] and ema_fast < ema_slow:
        signal = "Sell"
        indicators_used.extend(["RSI", "MACD", "EMA Cross"])
    else:
        signal = "Neutral"

    # --- Risk Management ---
    if signal == "Buy":
        entry_price = last_price
        stop_loss = entry_price - atr
        take_profit = entry_price + (atr * risk_reward_target)
    elif signal == "Sell":
        entry_price = last_price
        stop_loss = entry_price + atr
        take_profit = entry_price - (atr * risk_reward_target)
    else:
        entry_price, stop_loss, take_profit = last_price, None, None

    # --- Confidence Score ---
    confidence_score = 0.0
    if signal in ["Buy", "Sell"]:
        confidence_score = min(1.0, len(indicators_used) / 3)

    # --- Structured Output ---
    return {
        "pair": pair,
        "signal": signal,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "take_profit": round(take_profit, 2) if take_profit else None,
        "risk_reward_ratio": risk_reward_target if signal != "Neutral" else None,
        "indicators_used": indicators_used,
        "confidence_score": round(confidence_score, 2)
    }
