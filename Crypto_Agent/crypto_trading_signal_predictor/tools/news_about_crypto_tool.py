import os
import requests
from typing import Optional
from dotenv import load_dotenv
from agents import function_tool

load_dotenv()
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")

@function_tool
def news_about_crypto(coin: str, kind: str = "news") -> str:
    """
    Fetch latest crypto news from CryptoPanic for a specific coin/ticker.
    
    Args:
        coin (str): The coin ticker or keyword (e.g., "BTC", "Ethereum").
        kind (str, optional): Type of content to filter ("news" or "media").
    
    Returns:
        str: Nicely formatted news output.
    """
    print(f"Fetching news for coin: {coin.upper()} with kind: {kind}")
    if not CRYPTOPANIC_API_KEY:
        return "❌ API key missing. Please set CRYPTOPANIC_API_KEY in .env"

    base_url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": coin.upper(),
        "kind": kind,
        "public": "true"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return f"ℹ️ No news found for {coin.upper()}"

        formatted_articles = []
        for i, item in enumerate(data.get("results", []), start=1):
            title = item.get("title", "No title")
            link = item.get("url", "No URL")
            date = item.get("published_at", "No date")
            source = item.get("source", {}).get("title", "Unknown source")
            formatted_articles.append(
                f"{i}. **{title}**\n   📅 {date} | 📰 {source}\n   🔗 {link}"
            )

        return "\n\n".join(formatted_articles)

    except requests.exceptions.RequestException as e:
        return f"❌ Error fetching news: {e}"
