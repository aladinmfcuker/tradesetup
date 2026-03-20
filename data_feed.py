import json
import asyncio
import websockets
import requests
import pandas as pd
import logging
import numpy as np

class BinanceFeed:
    def __init__(self, symbol="xauusdt", timeframe="1h"):
        self.symbol = symbol.lower()
        self.timeframe = timeframe
        # Kline stream
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@kline_{self.timeframe}"
        # Level 2 Order Book stream (Top 10 bids/asks every 100ms)
        self.depth_ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth10@100ms"
        
        self.klines_url = "https://fapi.binance.com/fapi/v1/klines"
        self.df = pd.DataFrame()
        self.lock = asyncio.Lock() 
        
        self.order_book = {"bids": [], "asks": [], "imbalance": 0.0, "bid_vol": 0.0, "ask_vol": 0.0}
        self.ob_lock = asyncio.Lock()

    async def initialize(self):
        df = await asyncio.to_thread(self.fetch_historical_data)
        if df.empty:
            logging.warning("No valid historical data. Initializing with synthetic data for demonstration.")
            df = self._generate_dummy_data()
        async with self.lock:
            self.df = df

    def fetch_historical_data(self):
        logging.info(f"Fetching historical data for {self.symbol.upper()} from Binance Futures...")
        params = {"symbol": self.symbol.upper(), "interval": self.timeframe, "limit": 1500}
        
        try:
            response = requests.get(self.klines_url, params=params, timeout=10)
            data = response.json()
            
            if isinstance(data, dict) and "code" in data:
                logging.warning(f"Error fetching data from Binance: {data['msg']}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
                df[col] = df[col].astype(float)
            
            logging.info(f"Successfully loaded {len(df)} rows of historical data.")
            return df
        except Exception as e:
            logging.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    async def start_feed(self):
        logging.info(f"Connecting to Binance Kline WebSocket at {self.ws_url}")
        while True:
            try:
                async for websocket in websockets.connect(self.ws_url):
                    async for message in websocket:
                        data = json.loads(message)
                        if 'k' in data:
                            kline = data['k']
                            current_time = pd.to_datetime(kline['t'], unit='ms', utc=True)
                            
                            new_data = {
                                "timestamp": current_time,
                                "open": float(kline['o']),
                                "high": float(kline['h']),
                                "low": float(kline['l']),
                                "close": float(kline['c']),
                                "volume": float(kline['v']),
                                "quote_asset_volume": float(kline.get('q', 0)),
                                "number_of_trades": int(kline.get('n', 0)),
                                "taker_buy_base_asset_volume": float(kline.get('V', 0)),
                                "taker_buy_quote_asset_volume": float(kline.get('Q', 0))
                            }
                            
                            async with self.lock:
                                if not self.df.empty and self.df.iloc[-1]['timestamp'] == current_time:
                                    for key, value in new_data.items():
                                        self.df.at[self.df.index[-1], key] = value
                                else:
                                    self.df = pd.concat([self.df, pd.DataFrame([new_data])], ignore_index=True)
                                    if len(self.df) > 1000:
                                        self.df = self.df.iloc[-1000:].reset_index(drop=True)
            except websockets.ConnectionClosed:
                await asyncio.sleep(5)
            except Exception as e:
                await asyncio.sleep(5)

    async def start_depth_feed(self):
        logging.info(f"Connecting to Level 2 Order Book WebSocket at {self.depth_ws_url}")
        while True:
            try:
                async for websocket in websockets.connect(self.depth_ws_url):
                    async for message in websocket:
                        data = json.loads(message)
                        if 'b' in data and 'a' in data:
                            bids = data['b']
                            asks = data['a']
                            
                            # Calculate liquidity (Volume)
                            bid_vol = sum(float(b[1]) for b in bids)
                            ask_vol = sum(float(a[1]) for a in asks)
                            
                            # Imbalance formula: (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
                            # Positive = more limit buy orders (support), Negative = more limit sell orders (resistance)
                            total_vol = bid_vol + ask_vol
                            imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
                            
                            async with self.ob_lock:
                                self.order_book = {
                                    "bids": bids,
                                    "asks": asks,
                                    "imbalance": imbalance,
                                    "bid_vol": bid_vol,
                                    "ask_vol": ask_vol
                                }
            except websockets.ConnectionClosed:
                await asyncio.sleep(5)
            except Exception as e:
                await asyncio.sleep(5)

    async def get_dataframe(self):
        async with self.lock:
            return self.df.copy()

    async def get_order_book(self):
        async with self.ob_lock:
            return self.order_book.copy()

    def _generate_dummy_data(self):
        dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=500, freq='h')
        returns = np.random.normal(0, 2, 500)
        close = 2000 + np.cumsum(returns)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close - np.random.uniform(0, 2, 500),
            "high": close + np.random.uniform(0, 5, 500),
            "low": close - np.random.uniform(0, 5, 500),
            "close": close,
            "volume": np.random.uniform(100, 1000, 500)
        })
        return df
