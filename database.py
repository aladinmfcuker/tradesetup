import sqlite3
import json
import os
import logging
from datetime import datetime

DB_FILE = "trading_bot.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create trades table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT DEFAULT 'xauusdt',
        type TEXT NOT NULL,
        entry_price REAL NOT NULL,
        close_price REAL,
        target REAL NOT NULL,
        stop_loss REAL NOT NULL,
        pnl REAL,
        margin_used REAL,
        reason TEXT,
        ai_reasoning TEXT,
        timestamp TEXT NOT NULL,
        close_timestamp TEXT
    )
    ''')
    
    # Create portfolio history table for equity curve
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        balance REAL NOT NULL,
        open_pnl REAL NOT NULL,
        total_equity REAL NOT NULL
    )
    ''')
    
    # Create settings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    ''')
    
    # Create active positions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS active_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT DEFAULT 'xauusdt',
        type TEXT NOT NULL,
        entry_price REAL NOT NULL,
        size REAL NOT NULL,
        target REAL NOT NULL,
        stop_loss REAL NOT NULL,
        open_time TEXT NOT NULL,
        ai_reasoning TEXT,
        peak_price REAL,
        floor_price REAL
    )
    ''')
    
    # Check if symbol column exists in trades and active_positions (for existing DBs)
    cursor.execute("PRAGMA table_info(trades)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'symbol' not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN symbol TEXT DEFAULT 'xauusdt'")

    cursor.execute("PRAGMA table_info(active_positions)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'symbol' not in columns:
        cursor.execute("ALTER TABLE active_positions ADD COLUMN symbol TEXT DEFAULT 'xauusdt'")

    conn.commit()
    conn.close()

def migrate_from_json():
    if not os.path.exists("trade_history.json"):
        return
        
    try:
        with open("trade_history.json", "r") as f:
            data = json.load(f)
            
        if "history" not in data or not data["history"]:
            return
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if already migrated
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logging.info("Migrating old JSON trade history to SQLite...")
            for trade in data["history"]:
                cursor.execute('''
                INSERT INTO trades (symbol, type, entry_price, close_price, target, stop_loss, pnl, margin_used, reason, ai_reasoning, timestamp, close_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get("symbol", "xauusdt"),
                    trade.get("type"),
                    trade.get("entry_price"),
                    trade.get("close_price"),
                    trade.get("target"),
                    trade.get("stop_loss"),
                    trade.get("pnl"),
                    trade.get("margin_used", 0.0), # Default to 0 if not present in old data
                    trade.get("reason"),
                    trade.get("ai_reasoning", ""),
                    trade.get("timestamp"),
                    trade.get("timestamp") # Fallback for old data
                ))
            conn.commit()
            logging.info("Migration successful.")
            
            # Optionally backup/rename old file
            os.rename("trade_history.json", "trade_history_backup.json")
            
        conn.close()
    except Exception as e:
        logging.error(f"Failed to migrate JSON to SQLite: {e}")

class DatabaseManager:
    def __init__(self):
        init_db()
        migrate_from_json()
        
    def execute(self, query, params=()):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
        
    def fetch_all(self, query, params=()):
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
        
    def save_setting(self, key, value):
        self.execute('''
            INSERT INTO settings (key, value) 
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        ''', (key, json.dumps(value)))
        
    def get_settings(self):
        rows = self.fetch_all("SELECT * FROM settings")
        settings = {}
        for row in rows:
            try:
                settings[row['key']] = json.loads(row['value'])
            except json.JSONDecodeError:
                settings[row['key']] = row['value']
        return settings
        
    def save_trade(self, trade_data):
        return self.execute('''
            INSERT INTO trades (symbol, type, entry_price, close_price, target, stop_loss, pnl, margin_used, reason, ai_reasoning, timestamp, close_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get("symbol", "xauusdt"), trade_data.get("type"), trade_data.get("entry_price"), trade_data.get("close_price"),
            trade_data.get("target"), trade_data.get("stop_loss"), trade_data.get("pnl"), 
            trade_data.get("margin_used", 0), trade_data.get("reason"), trade_data.get("ai_reasoning", ""), 
            trade_data.get("timestamp"), datetime.now().isoformat()
        ))
        
    def get_trade_history(self, limit=50, symbol=None):
        if symbol:
            return self.fetch_all("SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?", (symbol, limit))
        return self.fetch_all("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))

    def save_portfolio_snapshot(self, balance, open_pnl, total_equity):
        # Only save a snapshot if the equity changed significantly or time passed? 
        # For now, we'll save it whenever requested.
        self.execute('''
            INSERT INTO portfolio_history (timestamp, balance, open_pnl, total_equity)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), balance, open_pnl, total_equity))
        
    def get_portfolio_history(self):
        # Group by hour/day if needed, but for now just return the latest 100 snapshots
        return self.fetch_all("SELECT * FROM portfolio_history ORDER BY id DESC LIMIT 100")

    def save_active_positions(self, positions, symbol='xauusdt'):
        self.execute("DELETE FROM active_positions WHERE symbol = ?", (symbol,))
        for pos in positions:
            self.execute('''
                INSERT INTO active_positions (symbol, type, entry_price, size, target, stop_loss, open_time, ai_reasoning, peak_price, floor_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pos.get("symbol", symbol), pos["type"], pos["entry_price"], pos["size"], pos["target"], pos["stop_loss"],
                pos["open_time"].isoformat() if hasattr(pos["open_time"], 'isoformat') else str(pos["open_time"]),
                pos.get("ai_reasoning", ""),
                pos.get("peak_price"), pos.get("floor_price")
            ))

    def get_active_positions(self, symbol=None):
        if symbol:
            rows = self.fetch_all("SELECT * FROM active_positions WHERE symbol = ?", (symbol,))
        else:
            rows = self.fetch_all("SELECT * FROM active_positions")
        positions = []
        for row in rows:
            pos = {
                "symbol": row.get("symbol", "xauusdt"),
                "type": row["type"],
                "entry_price": row["entry_price"],
                "size": row["size"],
                "target": row["target"],
                "stop_loss": row["stop_loss"],
                "open_time": datetime.fromisoformat(row["open_time"]) if "T" in row["open_time"] else row["open_time"],
                "ai_reasoning": row["ai_reasoning"]
            }
            if row["peak_price"] is not None:
                pos["peak_price"] = row["peak_price"]
            if row["floor_price"] is not None:
                pos["floor_price"] = row["floor_price"]
            positions.append(pos)
        return positions

db = DatabaseManager()
