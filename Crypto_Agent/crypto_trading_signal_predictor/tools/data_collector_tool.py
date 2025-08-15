from typing import List, Optional
import requests
from datetime import datetime, timedelta
from agents import function_tool
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from collections import defaultdict

load_dotenv()

COIN_GECKO_BASE_URL = os.getenv(
    "COIN_GECKO_BASE_URL",
    "https://api.coingecko.com/api/v3"
)

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    class Config:
        extra = 'forbid'
        
class GetCoinDetailsOutput(BaseModel):
    ohlcv: Optional[List[Candle]] = None
    error: Optional[str] = None

    class Config:
        extra = 'forbid'


# @function_tool
async def get_coin_details(input: str) -> GetCoinDetailsOutput:
    """
    Fetches OHLCV data for the last 14 days (converted to 1-hour candles)
    for a given crypto pair (e.g., 'BTC/USD') without requiring an API key.
    """

    print(f"Fetching 14 days OHLCV (converted to hourly) for: {input}")

    SYMBOL_TO_ID = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano", "XRP": "ripple",
        "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "matic-network", "LTC": "litecoin",
        "BCH": "bitcoin-cash", "LINK": "chainlink", "UNI": "uniswap", "XLM": "stellar",
        "AVAX": "avalanche-2", "ATOM": "cosmos", "TRX": "tron", "ALGO": "algorand",
        "VET": "vechain", "FIL": "filecoin", "ICP": "internet-computer", "AAVE": "aave",
        "THETA": "theta-network", "EOS": "eos", "XTZ": "tezos", "NEO": "neo",
        "KSM": "kusama", "ZEC": "zcash", "DASH": "dash"
    }

    try:
        base_symbol, quote_symbol = input.strip().upper().replace("-", "/").split("/")
        coin_id = SYMBOL_TO_ID.get(base_symbol)
        if not coin_id:
            return GetCoinDetailsOutput(error=f"Unsupported base symbol '{base_symbol}'.")
    except Exception:
        return GetCoinDetailsOutput(error= "Invalid input format. Use format like 'BTC/USDT'.")

    # Fetch OHLC data
    chart_url = f"{COIN_GECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": quote_symbol.lower(), "days": 14}

    try:
        res = requests.get(chart_url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if not data or not isinstance(data, list):
            return GetCoinDetailsOutput(error="No OHLCV data found from CoinGecko.")


        # Group candles by date to get exactly one daily candle
        daily_data = defaultdict(list)
        for ts, o, h, l, c in data:
            day_key = datetime.utcfromtimestamp(ts / 1000).date()
            daily_data[day_key].append((o, h, l, c))

        # Sort and prepare daily OHLC
        daily_candles = []
        for day in sorted(daily_data.keys()):
            o = daily_data[day][0][0]  # first open of the day
            h = max(c[1] for c in daily_data[day])  # max high
            l = min(c[2] for c in daily_data[day])  # min low
            c = daily_data[day][-1][3]  # last close of the day
            daily_candles.append((day, o, h, l, c))

        # Ensure we only have 14 days
        daily_candles = daily_candles[-14:]

        # Convert each daily candle to 24 hourly candles
        hourly_candles = []
        for day, o, h, l, c in daily_candles:
            day_start = datetime.combine(day, datetime.min.time())
            price_step = (c - o) / 24

            for hour in range(24):
                hourly_ts = day_start + timedelta(hours=hour)
                hourly_open = o + price_step * hour
                hourly_close = hourly_open + price_step
                hourly_high = max(hourly_open, hourly_close)
                hourly_low = min(hourly_open, hourly_close)

                hourly_candles.append(
                    Candle(
                        timestamp=hourly_ts,
                        open=hourly_open,
                        high=hourly_high,
                        low=hourly_low,
                        close=hourly_close,
                        volume=0.0
                    )
                )


        print(f"✅ Generated {len(hourly_candles)} hourly candles from daily data.")
        hourly_candles = hourly_candles[-150:]
        print(f"📉 Trimmed to {len(hourly_candles)} candles.")
        return GetCoinDetailsOutput(ohlcv=hourly_candles)

    except requests.RequestException as e:
        return GetCoinDetailsOutput(error=f"Network/API request failed: {str(e)}")
    except Exception as e:
        return GetCoinDetailsOutput(error=f"Unexpected error: {str(e)}")
