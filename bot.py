import os
import json
from google import genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Connect to Alpaca and Google Gemini (Using your free key)
trading_client = TradingClient(
    api_key=os.environ.get("ALPACA_API_KEY"),
    secret_key=os.environ.get("ALPACA_SECRET_KEY"),
    paper=True # Keeps trades on your paper account
)
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

ticker = "AAPL"
market_news = "Apple just announced a surprise product launch and stock is gaining volume."

prompt = f"""
Analyze this news for {ticker}: "{market_news}"
Should we BUY, SELL, or HOLD? 
Respond strictly in this JSON format, with no extra text or markdown code blocks:
{{"action": "BUY", "reason": "short explanation"}}
"""

print("Consulting Free Gemini AI...")
# We use gemini-2.5-flash because it is fast and 100% free
response = gemini_client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

# Clean and read the AI decision
raw_text = response.text.strip().replace("```json", "").replace("```", "")
decision = json.loads(raw_text)
print(f"Gemini's Decision: {decision['action']} | Reason: {decision['reason']}")

if decision["action"] in ["BUY", "SELL"]:
    side = OrderSide.BUY if decision["action"] == "BUY" else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=ticker,
        qty=1,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    print(f"Submitting {decision['action']} order to Alpaca Paper...")
    submitted_order = trading_client.submit_order(order_data=order)
    print(f"Success! Order ID: {submitted_order.id}")
else:
    print("Holding position.")
