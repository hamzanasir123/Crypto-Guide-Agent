from agents import function_tool
from tools.data_collector_tool import get_coin_details

NAME_TO_SYMBOL = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "ada": "ADA",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    # add more as needed
}

DEFAULT_QUOTE = "USDT"

def normalize_input(user_input: str) -> str:
    """
    Convert user input like 'ethereum' or 'ETH' to a proper SYMBOL/QUOTE format.
    """
    user_input = user_input.strip().lower().replace("-", "").replace(" ", "")
    if "/" in user_input:
        base, quote = user_input.split("/")
        base_sym = NAME_TO_SYMBOL.get(base, base.upper())
        return f"{base_sym}/{quote.upper()}"
    # Single token input
    base_sym = NAME_TO_SYMBOL.get(user_input)
    if base_sym:
        return f"{base_sym}/{DEFAULT_QUOTE}"
    # fallback if unknown
    return None




@function_tool
async def ohlcv_tool(input: str):
    pair = normalize_input(input)
    if not pair:
        return "Invalid input format. Use format like 'BTC/USDT' or just 'BTC'."
    print(f"Fetching predictions for: {pair}")
    response = await get_coin_details(pair)
    print("OHLCV Data Retrieved:")
    if response.ohlcv is None:
        print("Error:", response.error)
        return
    return response , pair
