import asyncio
from agents import function_tool
from tools.data_collector_tool import get_coin_details
from tools.apply_indicators_strategies_and_visualize import apply_indicators_strategies_and_visualize


@function_tool
async def get_predictions_tool(input: str):
    response = await get_coin_details(input)
    print("OHLCV Data Retrieved:")
    if response.ohlcv is None:
        print("Error:", response.error)
        return
    result = apply_indicators_strategies_and_visualize(response.ohlcv) 
    print("Indicators and Strategies Applied:")
    return result
