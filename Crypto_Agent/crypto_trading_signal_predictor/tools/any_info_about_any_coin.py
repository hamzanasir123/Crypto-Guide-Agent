import re
import requests
from agents import function_tool
from dotenv import load_dotenv
import os

from tools.data_collector_tool import GetCoinDetailsOutput

load_dotenv()

COIN_GECKO_BASE_URL = os.getenv(
    "COIN_GECKO_BASE_URL",
    "https://api.coingecko.com/api/v3"
)

SYMBOL_TO_ID = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano", "XRP": "ripple",
    "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "matic-network", "LTC": "litecoin",
    "BCH": "bitcoin-cash", "LINK": "chainlink", "UNI": "uniswap", "XLM": "stellar",
    "AVAX": "avalanche-2", "ATOM": "cosmos", "TRX": "tron", "ALGO": "algorand",
    "VET": "vechain", "FIL": "filecoin", "ICP": "internet-computer", "AAVE": "aave",
    "THETA": "theta-network", "EOS": "eos", "XTZ": "tezos", "NEO": "neo",
    "KSM": "kusama", "ZEC": "zcash", "DASH": "dash"
}

CURRENCY_CODES = {"USD", "USDT", "EUR", "GBP", "PKR", "INR", "JPY", "AUD", "CAD", "CHF", "CNY"}


def extract_pair_from_text(text: str):
    text = text.upper()

    # Match formats like BTC/USD, BTCUSDT, BTC-USD
    pair_match = re.search(r"\b([A-Z]{2,10})[\/\-]?([A-Z]{2,10})\b", text)
    if pair_match:
        base, quote = pair_match.groups()
        if base in SYMBOL_TO_ID:
            if quote in CURRENCY_CODES:
                return base, quote

    # Match single coin mentions like "bitcoin", "btc"
    for symbol, coin_id in SYMBOL_TO_ID.items():
        if symbol in text or coin_id.upper() in text:
            return symbol, "USD"  # default to USD if no currency found

    return None, None


@function_tool
async def any_info_about_any_coin(input: str):
    """
    Get real-time info about a cryptocurrency pair or coin from messy text input.
    Example inputs:
    - "What's BTC/USD price?"
    - "btc price"
    - "How much is ethereum to usd?"
    - "ethusdt"
    """

    base_symbol, quote_symbol = extract_pair_from_text(input)

    if not base_symbol:
        return {"error": "Could not detect a valid trading pair or coin name from your input."}

    coin_id = SYMBOL_TO_ID.get(base_symbol)
    if not coin_id:
        return GetCoinDetailsOutput(error=f"Unsupported base symbol '{base_symbol}'.")

    # Fetch current price
    price_url = f"{COIN_GECKO_BASE_URL}/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": quote_symbol.lower(),
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }

    try:
        res = requests.get(price_url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if coin_id not in data:
            return GetCoinDetailsOutput(error="No price data found from CoinGecko.")

        coin_data = data[coin_id]

        return {
            "symbol": f"{base_symbol}/{quote_symbol}",
            "price": coin_data.get(quote_symbol.lower()),
            "market_cap": coin_data.get(f"{quote_symbol.lower()}_market_cap"),
            "24h_volume": coin_data.get(f"{quote_symbol.lower()}_24h_vol"),
            "24h_change": coin_data.get(f"{quote_symbol.lower()}_24h_change")
        }

    except requests.RequestException as e:
        return GetCoinDetailsOutput(error=f"Network/API request failed: {str(e)}")
    except Exception as e:
        return GetCoinDetailsOutput(error=f"Unexpected error: {str(e)}")
