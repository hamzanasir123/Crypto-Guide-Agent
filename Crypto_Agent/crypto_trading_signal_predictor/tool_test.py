import asyncio
from datetime import datetime, timedelta, timezone
from tools.swing_trading_tool import Candle, swing_trading_tool


async def test_swing_trading_tool():
    # Sample input data
    symbol = "BTCUSDT"
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    intraday_ohlcv = [
        Candle(datetime=(now).isoformat(), open=50000, high=50500, low=49500, close=50200, volume=100),
        Candle(datetime=(now + timedelta(minutes=15)).isoformat(), open=50200, high=50600, low=49800, close=50400, volume=150),
    ]
    hourly_ohlcv = [
        Candle(datetime=(now + timedelta(hours=1)).isoformat(), open=50100, high=50700, low=49900, close=50550, volume=200),
    ]
    # Call the function
    result = swing_trading_tool(symbol=symbol, intraday_ohlcv=intraday_ohlcv, hourly_ohlcv=hourly_ohlcv)

    print(f"Symbol: {result.symbol}")
    print(f"Date: {result.date_local}")
    print(f"Timeframe Forward Hours: {result.timeframe_forward_hours}")
    print(f"Levels: {result.levels}")
    print(f"ATR 1H %: {result.atr_1h_pct}")
    print(f"Text: {result.text}")


asyncio.run(test_swing_trading_tool())