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

# ==============================================================================
# TRADING ENGINE CONFIGURATION
# ==============================================================================

# TOGGLE ENVIRONMENT: Set to True to trade with REAL MONEY. Set to False for Paper.
LIVE_TRADING = False  

# Execution Frequency (in seconds)
RUN_INTERVAL_SECONDS = 60  # 5 minutes

# ==============================================================================
# INJECTABLE TRADING STRATEGY SETUP
# Modify this dictionary to update your strategy logic without changing API code.
# ==============================================================================
STRATEGY_CONFIG = {
    "strategy_name": "AI Sentiment & Momentum Breakout",
    "target_tickers": ["AAPL", "NVDA", "MSFT"],
    "max_position_size_shares": 1,
    "risk_tolerance": "MODERATE",  # CONSERVATIVE, MODERATE, AGGRESSIVE
    "rules": [
        "Only buy if there is recent, verifiable positive news momentum or earnings upgrades.",
        "Sell immediately if major negative press, bad earnings, or regulatory action is reported.",
        "If news is mixed or neutral, hold current position or take no action.",
        "Do NOT buy if stock price has already dropped over 5% today due to negative catalysts."
    ]
}


class DecisionSchema(BaseModel):
    ticker: str = Field(description="The ticker symbol evaluated")
    action: str = Field(description="Must be strictly BUY, SELL, or HOLD")
    reason: str = Field(description="Detailed explanation based on news and strategy rules")


def run_trading_cycle(trading_client: TradingClient, gemini_client: genai.Client):
    mode_label = "LIVE TRADING" if LIVE_TRADING else "PAPER TRADING"
    print(f"\n==================================================")
    print(f"--- Running Trading Cycle [{mode_label}] ---")
    print(f"==================================================")
    
    # 1. Market Hours Check
    try:
        clock = trading_client.get_clock()
        if not clock.is_open:
            print(f"Market is closed. Next market open at: {clock.next_open}")
            return
    except Exception as e:
        print(f"Error checking market clock: {e}")
        return

    # 2. Get Current Owned Portfolio
    try:
        positions = trading_client.get_all_positions()
        owned_positions = {p.symbol: float(p.qty) for p in positions}
        print(f"Current Positions Held: {list(owned_positions.keys()) if owned_positions else 'None'}")
    except Exception as e:
        print(f"Error checking portfolio positions: {e}")
        return

    # 3. Process Tickers under Strategy Parameters
    for ticker in STRATEGY_CONFIG["target_tickers"]:
        has_position = ticker in owned_positions
        current_qty = owned_positions.get(ticker, 0)
        
        print(f"\nAnalyzing [{ticker}] | Currently Held: {current_qty} shares")

        # Dynamic System Prompt injecting user rules
        prompt = f"""
        System Role: You are an execution engine enforcing a strict trading strategy.
        
        STRATEGY CONSTRAINTS:
        - Strategy Name: {STRATEGY_CONFIG['strategy_name']}
        - Risk Tolerance: {STRATEGY_CONFIG['risk_tolerance']}
        - Core Rules to Enforce: {json.dumps(STRATEGY_CONFIG['rules'])}
        
        CURRENT MARKET POSITION:
        - Target Stock: {ticker}
        - Currently Holding: {has_position} ({current_qty} shares)
        
        TASK:
        Search Google for today's live breaking news, price trends, and financial reports for {ticker}.
        Evaluate whether the news matches the rules outlined in the strategy constraints above.
        
        DECISION DIRECTIVE:
        - If holding position: Decide between SELL or HOLD.
        - If NOT holding position: Decide between BUY or HOLD.
        """

        try:
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

            decision = json.loads(response.text)
            action = decision.get("action", "HOLD").upper()
            reason = decision.get("reason", "No reason provided.")
            
            print(f"[{ticker}] AI Decision: {action}")
            print(f"Reason: {reason}")

            # 4. Order Execution Engine
            if action in ["BUY", "SELL"]:
                side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
                qty = STRATEGY_CONFIG["max_position_size_shares"]

                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                print(f"--> SUBMITTING {mode_label} ORDER: {action} {qty} share(s) of {ticker}...")
                submitted_order = trading_client.submit_order(order_data=order)
                print(f"--> SUCCESS! Order ID: {submitted_order.id}")
            else:
                print(f"--> NO ACTION TAKEN for {ticker}.")

        except APIError as e:
            print(f"Gemini API Error for {ticker}: {e}")
        except AlpacaAPIError as e:
            print(f"Alpaca Order Submission Failed for {ticker}: {e}")
        except Exception as e:
            print(f"Unexpected Error evaluating {ticker}: {e}")


def main():
    # Load and validate API keys
    google_key = os.environ.get("GEMINI_API_KEY")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not all([google_key, alpaca_key, alpaca_secret]):
        print("CRITICAL ERROR: Missing required environment variables (GEMINI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY).")
        sys.exit(1)

    # Initialize Alpaca Client with Live/Paper Toggle
    trading_client = TradingClient(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
        paper=not LIVE_TRADING  # paper=False triggers real-money trades
    )
    gemini_client = genai.Client(api_key=google_key)

    mode_text = "LIVE REAL-MONEY TRADING" if LIVE_TRADING else "PAPER TRADING (Simulated)"
    print(f"Bot successfully started in **{mode_text}** mode.")
    print(f"Running strategy '{STRATEGY_CONFIG['strategy_name']}' every {RUN_INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            run_trading_cycle(trading_client, gemini_client)
        except Exception as e:
            print(f"Execution Loop Error: {e}")
            
        time.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
