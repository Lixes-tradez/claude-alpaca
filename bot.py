import os
import json
from google import genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Pull the keys out of the hidden cloud system vault
GOOGLE_KEY = os.environ.get("GEMINI_API_KEY")
ALPACA_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY")

# 2. Connect to Alpaca and Google Gemini using those keys
trading_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=True  # Keeps trades locked safely to your free paper account
)
gemini_client = genai.Client(api_key=GOOGLE_KEY)

# 3. Setup the scenario for the AI to judge
ticker = "AAPL"
market_news = "Apple just announced a breakthrough AI product and stock volume is surging."

prompt = f"""
Analyze this news for {ticker}: "{market_news}"
Should we BUY, SELL, or HOLD? 
Respond strictly in this JSON format, with no extra conversational text or formatting:
{{"action": "BUY", "reason": "short explanation"}}
"""

print("Consulting Free Gemini AI...")
response = gemini_client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

# 4. Clean up the AI text data response
raw_text = response.text.strip().replace("```json", "").replace("```", "")
decision = json.loads(raw_text)
print(f"Gemini's Decision: {decision['action']} | Reason: {decision['reason']}")

# 5. Place the trade onto your Alpaca Paper Account
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
    print("Holding position. No trade placed.")
