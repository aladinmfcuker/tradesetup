import requests
import logging

def get_fear_and_greed_index():
    """
    Fetches the latest Crypto Fear & Greed Index from alternative.me
    Returns a dict with 'value' (0-100) and 'classification' (e.g. 'Extreme Greed')
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                latest = data["data"][0]
                return {
                    "value": int(latest["value"]),
                    "classification": latest["value_classification"]
                }
    except Exception as e:
        logging.error(f"Failed to fetch Fear & Greed Index: {e}")
        
    return {"value": 50, "classification": "Neutral (Default)"}

def get_market_sentiment():
    """
    Combines various sentiment metrics (can be expanded later to include news APIs).
    For now, returns the Fear and Greed index.
    """
    fng = get_fear_and_greed_index()
    return fng
