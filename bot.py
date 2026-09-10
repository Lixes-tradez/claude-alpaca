import os
import json
from anthropic import Anthropic
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Connect to Alpaca and Claude using secure GitHub environment variables
trading_client = TradingClient(
    api_key=os.environ.get("ALPACA_API_KEY"),
    secret_key=os.environ.get("ALPACA_SECRET_KEY"),
    paper=True # Keeps trades on your paper account
)
claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ticker = "AAPL"
market_news = "Apple just announced a surprise product launch and stock is gaining volume."

prompt = f"""
Analyze this news for {ticker}: "{market_news}"
Should we BUY, SELL, or HOLD? 
Respond strictly in this JSON format, with no extra text:
{{"action": "BUY", "reason": "short explanation"}}
"""

print("Consulting Claude...")
response = claude_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=150,
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)

decision = json.loads(response.content.text)
print(f"Claude's Decision: {decision['action']} | Reason: {decision['reason']}")

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

