from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Hist': hist})

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def apply_indicators_strategies_and_visualize(ohlcv: List[Candle]) -> Dict[str, Any]:
    df = pd.DataFrame([c.dict() for c in ohlcv])
    df.set_index('timestamp', inplace=True)

    # Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    macd_df = calculate_macd(df['close'])
    df = pd.concat([df, macd_df], axis=1)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['close'])

    # Strategy 1: SMA crossover
    df['SMA_signal'] = 0
    df.loc[df['SMA_5'] > df['SMA_10'], 'SMA_signal'] = 1
    df.loc[df['SMA_5'] < df['SMA_10'], 'SMA_signal'] = -1

    # Strategy 2: EMA crossover
    df['EMA_signal'] = 0
    df.loc[df['EMA_5'] > df['EMA_10'], 'EMA_signal'] = 1
    df.loc[df['EMA_5'] < df['EMA_10'], 'EMA_signal'] = -1

    # Strategy 3: RSI strategy
    df['RSI_signal'] = 0
    df.loc[df['RSI_14'] < 30, 'RSI_signal'] = 1    # Oversold -> Buy
    df.loc[df['RSI_14'] > 70, 'RSI_signal'] = -1   # Overbought -> Sell

    # Strategy 4: MACD crossover
    df['MACD_signal'] = 0
    df['MACD_signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
    df['MACD_cross'] = df['MACD_signal'].diff()
    df['MACD_crossover_signal'] = 0
    df.loc[df['MACD_cross'] == 2, 'MACD_crossover_signal'] = 1
    df.loc[df['MACD_cross'] == -2, 'MACD_crossover_signal'] = -1

    # Strategy 5: Bollinger Bands
    df['BB_signal'] = 0
    # Buy if price crosses below lower band (from above)
    df.loc[(df['close'] < df['BB_Lower']) & (df['close'].shift(1) >= df['BB_Lower'].shift(1)), 'BB_signal'] = 1
    # Sell if price crosses above upper band (from below)
    df.loc[(df['close'] > df['BB_Upper']) & (df['close'].shift(1) <= df['BB_Upper'].shift(1)), 'BB_signal'] = -1

    # Combine all strategy signals
    df['combined_signal'] = df['SMA_signal'] + df['EMA_signal'] + df['RSI_signal'] + df['MACD_crossover_signal'] + df['BB_signal']

    def interpret_signal(x):
        if x > 0:
            return 'Buy'
        elif x < 0:
            return 'Sell'
        else:
            return 'Hold'

    df['signal_meaning'] = df['combined_signal'].apply(interpret_signal)

    # Alert messages for signal changes
    df['previous_signal'] = df['signal_meaning'].shift(1)
    df['alert'] = ''
    for idx, row in df.iterrows():
        if row['signal_meaning'] != row['previous_signal']:
            if row['signal_meaning'] == 'Buy':
                df.at[idx, 'alert'] = f"Buy signal generated at {idx}"
            elif row['signal_meaning'] == 'Sell':
                df.at[idx, 'alert'] = f"Sell signal generated at {idx}"

    # Prepare signals over time for output
    signals_over_time = df[['close', 'SMA_signal', 'EMA_signal', 'RSI_signal', 'MACD_crossover_signal', 'BB_signal', 'combined_signal', 'signal_meaning', 'alert']].dropna().to_dict(orient='index')

    latest = df.iloc[-1]
    latest_summary = {
        "latest_close": latest['close'],
        "SMA_5": latest['SMA_5'],
        "SMA_10": latest['SMA_10'],
        "EMA_5": latest['EMA_5'],
        "EMA_10": latest['EMA_10'],
        "RSI_14": latest['RSI_14'],
        "MACD": latest['MACD'],
        "Signal_Line": latest['Signal'],
        "BB_Middle": latest['BB_Middle'],
        "BB_Upper": latest['BB_Upper'],
        "BB_Lower": latest['BB_Lower'],
        "SMA_signal": latest['SMA_signal'],
        "EMA_signal": latest['EMA_signal'],
        "RSI_signal": latest['RSI_signal'],
        "MACD_crossover_signal": latest['MACD_crossover_signal'],
        "BB_signal": latest['BB_signal'],
        "combined_signal": latest['combined_signal'],
        "signal_meaning": latest['signal_meaning'],
        "alert": latest['alert'],
    }

    # Visualization helper (returns base64 string or shows plot if running locally)
    def plot_indicators_signals(df):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,10), sharex=True)

        # Price + Bollinger Bands + Buy/Sell markers
        ax1.plot(df.index, df['close'], label='Close Price', color='black')
        ax1.plot(df.index, df['BB_Middle'], label='BB Middle', color='blue', linestyle='--')
        ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', linestyle='--')
        ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', linestyle='--')

        # Mark Buy/Sell signals on price chart
        buy_signals = df[df['signal_meaning'] == 'Buy']
        sell_signals = df[df['signal_meaning'] == 'Sell']
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

        ax1.set_title('Price with Bollinger Bands and Buy/Sell Signals')
        ax1.legend()

        # RSI plot
        ax2.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_title('RSI Indicator')
        ax2.legend()

        # MACD plot
        ax3.plot(df.index, df['MACD'], label='MACD Line', color='blue')
        ax3.plot(df.index, df['Signal'], label='Signal Line', color='orange')
        ax3.bar(df.index, df['Hist'], label='Histogram', color='gray')
        ax3.set_title('MACD Indicator')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    # Optionally, call plot_indicators_signals(df) here if you want to see plot during run.

    return {
        "latest_summary": latest_summary,
        "signals_over_time": signals_over_time,
        "plot_function": plot_indicators_signals ,
        "dataframe" : df 
    }
