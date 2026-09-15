import os
import json
import sys
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError as AlpacaAPIError


# Define schema for Gemini's structured output
class DecisionSchema(BaseModel):
    action: str = Field(description="Must be strictly BUY, SELL, or HOLD")
    reason: str = Field(description="Short explanation for the decision")


def main():
    # 1. Validate environment variables upfront
    google_key = os.environ.get("GEMINI_API_KEY")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

    missing_keys = []
    if not google_key: missing_keys.append("GEMINI_API_KEY")
    if not alpaca_key: missing_keys.append("ALPACA_API_KEY")
    if not alpaca_secret: missing_keys.append("ALPACA_SECRET_KEY")

    if missing_keys:
        print(f"Error: Missing required environment variables: {', '.join(missing_keys)}")
        sys.exit(1)

    # 2. Initialize Clients
    try:
        trading_client = TradingClient(
            api_key=alpaca_key,
            secret_key=alpaca_secret,
            paper=True
        )
        gemini_client = genai.Client(api_key=google_key)
    except Exception as e:
        print(f"Failed to initialize API clients: {e}")
        sys.exit(1)

    # 3. Setup context for the AI analysis
    ticker = "AAPL"
    market_news = "Apple just announced a breakthrough AI product and stock volume is surging."
    prompt = f'Analyze this news for ticker {ticker}: "{market_news}". Decide whether to BUY, SELL, or HOLD.'

    # 4. Request structured output from Gemini
    print("Consulting Gemini AI...")
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DecisionSchema,
                temperature=0.1,  # Low temperature for deterministic decisions
            )
        )
        
        # Safe JSON loading backed by Pydantic response schema
        decision = json.loads(response.text)
        action = decision.get("action", "HOLD").upper()
        reason = decision.get("reason", "No reason provided.")
        
        print(f"Gemini's Decision: {action} | Reason: {reason}")

    except APIError as e:
        print(f"Gemini API Exception: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during AI analysis: {e}")
        sys.exit(1)

    # 5. Execute Order on Alpaca
    if action in ["BUY", "SELL"]:
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=ticker,
            qty=1,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        try:
            print(f"Submitting {action} order for {ticker} to Alpaca Paper...")
            submitted_order = trading_client.submit_order(order_data=order)
            print(f"Success! Order ID: {submitted_order.id}")
        except AlpacaAPIError as e:
            print(f"Alpaca API execution failed: {e}")
        except Exception as e:
            print(f"Unexpected error executing trade: {e}")
    else:
        print("Action is HOLD or unmapped. No trade executed.")


if __name__ == "__main__":
    main()
