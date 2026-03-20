import yfinance as yf
import pandas as pd
import logging

def get_macro_data():
    """
    Fetches US Dollar Index proxy (UUP) and US10Y (US 10-Year Treasury Yield).
    Gold is highly inversely correlated to the Dollar and Yields.
    """
    macro_data = {"dxy": None, "us10y": None}
    
    try:
        # Use UUP as a highly reliable ETF proxy for the DXY
        dxy_data = yf.download("UUP", period="1d", progress=False)
        if not dxy_data.empty:
            if isinstance(dxy_data.columns, pd.MultiIndex):
                 macro_data["dxy"] = float(dxy_data['Close']['UUP'].dropna().iloc[-1])
            else:
                 macro_data["dxy"] = float(dxy_data['Close'].dropna().iloc[-1])
    except Exception as e:
        logging.warning(f"Failed to fetch DXY proxy (UUP): {e}")

    try:
        # ^TNX = US 10-Year Treasury Yield
        us10y_data = yf.download("^TNX", period="1d", progress=False)
        if not us10y_data.empty:
            if isinstance(us10y_data.columns, pd.MultiIndex):
                 macro_data["us10y"] = float(us10y_data['Close']['^TNX'].dropna().iloc[-1])
            else:
                 macro_data["us10y"] = float(us10y_data['Close'].dropna().iloc[-1])
    except Exception as e:
        logging.warning(f"Failed to fetch US10Y (^TNX): {e}")

    return macro_data
