import os
import json
import sys
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError as AlpacaAPIError

# Interval between trading checks (in seconds)
RUN_INTERVAL_SECONDS = 100000  # 15 minutes


class DecisionSchema(BaseModel):
    action: str = Field(description="Must be strictly BUY, SELL, or HOLD")
    reason: str = Field(description="Short explanation based on live market news")


def run_trading_cycle(trading_client: TradingClient, gemini_client: genai.Client):
    print("\n--- Starting New Trading Cycle ---")
    
    # 1. Safeguard: Check if the US Stock Market is open
    try:
        clock = trading_client.get_clock()
        if not clock.is_open:
            print(f"Market is closed. Next market open at: {clock.next_open}")
            return
    except Exception as e:
        print(f"Error checking market clock: {e}")
        return

    ticker = "AAPL"
    
    # 2. Portfolio Check: See if we currently hold shares of ticker
    has_position = False
    try:
        positions = trading_client.get_all_positions()
        owned_tickers = [p.symbol for p in positions]
        if ticker in owned_tickers:
            has_position = True
            print(f"Current Status: Currently holding {ticker} in portfolio.")
        else:
            print(f"Current Status: Not holding {ticker}.")
    except Exception as e:
        print(f"Error checking current positions: {e}")
        return

    # 3. Dynamic Prompt tailored to live search
    if has_position:
        prompt = (
            f"Search Google for today's latest financial news and price action for {ticker}. "
            f"I currently hold a position in {ticker}. Based on live web news, decide whether to SELL or HOLD."
        )
    else:
        prompt = (
            f"Search Google for today's latest financial news and price action for {ticker}. "
            f"I do not hold a position in {ticker}. Based on live web news, decide whether to BUY or HOLD."
        )

    # 4. Request decision using Google Search Grounding
    print(f"Fetching live market news for {ticker} via Gemini Google Search...")
    try:
        # Enable Google Search Grounding tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_schema=DecisionSchema,
            temperature=0.1,
        )

        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=config
        )
        
        # Display search metadata/queries executed if available
        candidates = response.candidates
        if candidates and candidates[0].grounding_metadata:
            search_queries = candidates[0].grounding_metadata.web_search_queries
            if search_queries:
                print(f"Executed Web Searches: {', '.join(search_queries)}")

        decision = json.loads(response.text)
        action = decision.get("action", "HOLD").upper()
        reason = decision.get("reason", "No reason provided.")
        print(f"AI Decision: {action} | Reason: {reason}")

    except APIError as e:
        print(f"Gemini API Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during AI evaluation: {e}")
        return

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
            print(f"Submitting {action} order for 1 share of {ticker}...")
            submitted_order = trading_client.submit_order(order_data=order)
            print(f"Order Executed Successfully! Order ID: {submitted_order.id}")
        except AlpacaAPIError as e:
            print(f"Alpaca Order Submission Failed: {e}")
    else:
        print("Decision is HOLD. No trade placed.")


def main():
    # Load and validate credentials
    google_key = os.environ.get("GEMINI_API_KEY")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not all([google_key, alpaca_key, alpaca_secret]):
        print("Error: Missing one or more required environment variables.")
        sys.exit(1)

    # Initialize API Clients
    trading_client = TradingClient(api_key=alpaca_key, secret_key=alpaca_secret, paper=True)
    gemini_client = genai.Client(api_key=google_key)

    print(f"Bot initialized with Live Web Search. Running loop every {RUN_INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to stop.")

    # Continuous execution loop
    while True:
        try:
            run_trading_cycle(trading_client, gemini_client)
        except Exception as e:
            print(f"Unexpected error in execution loop: {e}")
            
        time.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
