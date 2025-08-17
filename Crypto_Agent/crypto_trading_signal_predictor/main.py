import asyncio
import os
from agents import Agent, RunConfig, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, enable_verbose_stdout_logging
from dotenv import load_dotenv
from openai import AsyncOpenAI, InternalServerError
from tools.any_info_about_any_coin_tool import any_info_about_any_coin
from tools.get_predictions_tool import get_predictions_tool
from tools.news_about_crypto_tool import news_about_crypto
from tools.ohlcv_tool import ohlcv_tool
from tools.swing_trading_tool import swing_trading_tool

# --- Load environment variables ---
load_dotenv()
set_tracing_disabled(True)
# enable_verbose_stdout_logging()

with open("Instructions/main_agents.md", "r") as file:
    main_agent_instructions = file.read()

# --- Gemini API Keys ---
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3")
]
current_key_index = 0

if not GEMINI_KEYS or GEMINI_KEYS == [""]:
    raise ValueError("No Gemini API keys found in .env file.")

# --- Helper Functions ---
coin_map = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "dogecoin": "DOGE",
    "litecoin": "LTC",
    "tether": "USDT",
    "pkr": "PKR",
    "usd": "USD"
}

def detect_pair(text):
    words = text.lower().split()
    detected = [coin_map[w] for w in words if w in coin_map]
    if len(detected) >= 2:
        return f"{detected[0]}/{detected[1]}"
    elif len(detected) == 1:
        return f"{detected[0]}/USD"
    else:
        return None

def detect_intent_and_pair(user_input: str):
    text = user_input.lower()
    general_keywords = ["hello", "hi", "how are you", "good morning", "good evening", "thanks", "thank you"]
    if any(k in text for k in general_keywords):
        return "general", None

    trading_pair = detect_pair(text)

    if any(word in text for word in ["predict", "forecast", "future", "signal"]):
        intent = "prediction"
    elif any(word in text for word in ["price", "rate", "value", "worth", "current"]):
        intent = "price"
    elif any(word in text for word in ["news", "update", "article", "media"]):
        intent = "news"
    elif trading_pair:
        intent = "price"
    else:
        intent = "general"

    return intent, trading_pair

# --- Conversation Memory ---
conversation_memory = []

async def handle_user_message(user_input):
    global current_key_index, conversation_memory

    conversation_memory.append({"sender": "user", "text": user_input})
    intent, pair = detect_intent_and_pair(user_input)

    if pair:
        print(f"✅ Detected Pair: {pair}")

    while current_key_index < len(GEMINI_KEYS):
        api_key = GEMINI_KEYS[current_key_index].strip()
        external_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        model = OpenAIChatCompletionsModel(
            model="gemini-1.5-flash",
            openai_client=external_client
        )

        config = RunConfig(
            model=model,
            model_provider=external_client,
            tracing_disabled=True
        )

        agent = Agent(
            name="Crypto Trading Signal Predictor",
            instructions=main_agent_instructions,
            tools=[
                get_predictions_tool,
                any_info_about_any_coin,
                news_about_crypto,
                swing_trading_tool,
                ohlcv_tool
            ],
            model=model
        )

        try:
            conversation_text = "\n".join([f"{m['sender']}: {m['text']}" for m in conversation_memory])

            print("🤖 Thinking...")
            result = await Runner.run(
                starting_agent=agent,
                input=f"{conversation_text}\nIntent: {intent}\nPair: {pair}",
                run_config=config
            )

            # bot_response = result.output_text if hasattr(result, "output_text") else str(result)
            print(f"🤖 {result.final_output}")

            conversation_memory.append({"sender": "bot", "text": result.final_output})
            return

        except Exception as e:
            error_text = str(e).lower()
            if "resource_exhausted" in error_text or "quota" in error_text or "rate limit" in error_text:
                print(f"[Error] Gemini quota reached for key #{current_key_index + 1}")
                current_key_index += 1
                if current_key_index >= len(GEMINI_KEYS):
                    print("❌ All Gemini keys exhausted.")
                    return
                print(f"🔄 Switching to Gemini key #{current_key_index + 1}...")
            elif isinstance(e, InternalServerError) and "503" in str(e):
                print("⚠️ Model overloaded. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            else:
                print(f"❌ Error: {str(e)}")
                return

async def main():
    print("💬 Hi! I'm your Crypto Trading Assistant. Ask me about any coin!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        await handle_user_message(user_input)

if __name__ == "__main__":
    asyncio.run(main())
