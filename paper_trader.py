import logging
import json
import os
import requests
from datetime import datetime
from database import db
from app_state import state

def send_discord_alert(message):
    webhook_url = state.settings.get("discord_webhook")
    if webhook_url:
        try:
            payload = {"content": message}
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logging.error(f"Failed to send Discord alert: {e}")

class PaperTrader:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.current_price = 0.0
        self.current_atr = 0.0
        
        self._load_history()
        self._load_active_positions()

    def _load_active_positions(self):
        try:
            self.positions = db.get_active_positions()
            # Calculate the amount of balance locked in margin for active positions
            margin_in_use = sum([pos["entry_price"] * pos["size"] for pos in self.positions])
            # Balance should be updated to reflect locked margin
            self.balance -= margin_in_use
            if self.positions:
                logging.info(f"Loaded {len(self.positions)} active positions from DB.")
        except Exception as e:
            logging.error(f"Failed to load active positions from DB: {e}")

    def _save_active_positions(self):
        try:
            db.save_active_positions(self.positions)
        except Exception as e:
            logging.error(f"Failed to save active positions to DB: {e}")

    def update_atr(self, atr):
        self.current_atr = atr

    def _load_history(self):
        try:
            self.trade_history = db.get_trade_history(limit=50)
            
            # Recalculate balance from history to be accurate
            for trade in self.trade_history:
                self.balance += float(trade.get("pnl", 0))
                
            logging.info(f"Loaded {len(self.trade_history)} past trades for AI learning. Current simulated balance: ${self.balance:.2f}")
        except Exception as e:
            logging.error(f"Failed to load trade history from DB: {e}")

    def update_price(self, current_price):
        self.current_price = current_price
        self._check_positions()
        
        # Save a snapshot of the portfolio equity occasionally
        open_pnl = sum([self._calculate_unrealized_pnl(p) for p in self.positions])
        total_equity = self.balance + open_pnl
        
        # Adding a simple time throttle so we don't spam the DB every tick
        if not hasattr(self, '_last_snapshot_time') or (datetime.now() - self._last_snapshot_time).total_seconds() > 3600:
            db.save_portfolio_snapshot(self.balance, open_pnl, total_equity)
            self._last_snapshot_time = datetime.now()

    def _calculate_unrealized_pnl(self, pos):
        if pos["type"] == "LONG":
            return (self.current_price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - self.current_price) * pos["size"]

    def process_prediction(self, prediction_json, latest_data=None, margin_fraction=0.20):
        if self.current_price == 0.0 or latest_data is None:
            return

        try:
            if isinstance(prediction_json, str):
                prediction = json.loads(prediction_json)
            else:
                prediction = prediction_json

            if "error" in prediction:
                logging.warning(f"Skipping trade evaluation due to prediction error: {prediction['error']}")
                return

            pred_4h = prediction.get("prediction", {}).get("4h", {})
            direction = pred_4h.get("direction", "").lower()
            target_str = pred_4h.get("target")
            stop_loss_str = pred_4h.get("stop_loss")
            reasoning = prediction.get("reasoning", "No reasoning provided.")
            
            # Extract quant metrics
            z_score = latest_data.get('z_score', 0)
            vol_imbalance = latest_data.get('vol_imbalance_ema', 0)
            atr = latest_data.get('atr', 0)
            
            # Dynamic Risk Sizing Based on Confidence
            confidence = 50 # Default to 50
            try:
                conf_val = prediction.get("confidence", "50")
                if isinstance(conf_val, str) and "-" in conf_val: # Handle ranges like "70-80"
                    conf_val = conf_val.split("-")[0]
                confidence = float(conf_val)
            except (ValueError, TypeError):
                logging.warning(f"Could not parse confidence: {prediction.get('confidence')}. Defaulting to 50.")
            
            # Extract ensemble size multiplier if available from ai_agent.py (V4)
            ensemble_data = prediction.get("_ensemble", {})
            size_multiplier = ensemble_data.get("size_multiplier", 1.0)
            
            # Multiply LLM confidence by the mathematical quant signal multiplier
            effective_confidence = confidence * size_multiplier

            if effective_confidence < 30: # Use adjusted confidence for rejection
                logging.info(f"Effective AI Confidence too low ({effective_confidence:.1f}% after {size_multiplier}x regime multiplier). Skipping trade.")
                return

            # Institutional Quant Execution Filter
            # Only trade if the AI bias aligns with statistical reality
            quant_approved = False
            quant_reason = ""
            
            if "bullish" in direction or "long" in direction:
                # Require price not to be extremely overbought (Z-Score < 2.0)
                # and ideally some buying pressure
                if z_score < 2.0:
                    quant_approved = True
                    quant_reason = f"Z-Score {z_score:.2f} < 2.0 (Not severely overbought)"
                else:
                    logging.info(f"Quant Filter Rejected LONG: Z-Score {z_score:.2f} is too high (Overbought). Requires < 2.0.")
                    
            elif "bearish" in direction or "short" in direction:
                # Require price not to be extremely oversold (Z-Score > -2.0)
                if z_score > -2.0:
                    quant_approved = True
                    quant_reason = f"Z-Score {z_score:.2f} > -2.0 (Not severely oversold)"
                else:
                    logging.info(f"Quant Filter Rejected SHORT: Z-Score {z_score:.2f} is too low (Oversold). Requires > -2.0.")

            if not quant_approved:
                return

            if not direction or not target_str or not stop_loss_str:
                return

            try:
                target = float(str(target_str).replace(',', ''))
                stop_loss = float(str(stop_loss_str).replace(',', ''))
            except ValueError:
                logging.warning(f"Failed to parse target/stop_loss as float: TP={target_str}, SL={stop_loss_str}")
                return

            position_type = None
            if "bullish" in direction or "long" in direction:
                position_type = "LONG"
                if target <= self.current_price or stop_loss >= self.current_price:
                    logging.warning(f"Invalid LONG setup parameters from AI: Price={self.current_price}, TP={target}, SL={stop_loss}")
                    return
            elif "bearish" in direction or "short" in direction:
                position_type = "SHORT"
                if target >= self.current_price or stop_loss <= self.current_price:
                    logging.warning(f"Invalid SHORT setup parameters from AI: Price={self.current_price}, TP={target}, SL={stop_loss}")
                    return

            if position_type:
                # ANTI-PYRAMIDING LOGIC: Check if we already have an open position in this direction
                existing_position = next((p for p in self.positions if p["type"] == position_type), None)
                if existing_position:
                    logging.info(f"Anti-Pyramiding: Already in an active {position_type} position. Ignoring new signal to prevent over-exposure.")
                    return
                    
                # Check for hedging (long and short at the same time). For now, we block it to keep logic clean.
                opposite_type = "SHORT" if position_type == "LONG" else "LONG"
                hedged_position = next((p for p in self.positions if p["type"] == opposite_type), None)
                if hedged_position:
                     logging.info(f"Anti-Hedging: Cannot open {position_type} while an active {opposite_type} position exists.")
                     return

                # Combine reasoning
                full_reasoning = f"AI: {reasoning} | Quant: {quant_reason} | Ensemble Multiplier: {size_multiplier}x"
                self._open_position(position_type, target, stop_loss, full_reasoning, atr=atr, confidence=effective_confidence)

        except Exception as e:
            logging.error(f"Error parsing prediction for trading: {e}")

    def _open_position(self, position_type, target, stop_loss, reasoning, margin_fraction=None, atr=None, confidence=50):
        # Calculate risk distance
        risk_distance = abs(self.current_price - stop_loss)
        
        # Fallback to ATR if stop loss is too tight or weird
        if atr and atr > 0:
            if risk_distance < (atr * 0.5):
                risk_distance = atr * 1.5 # Ensure minimum stop distance
                
        if risk_distance <= 0:
            logging.warning("Invalid risk distance for sizing. Aborting trade.")
            return

        if state.settings.get("dynamic_sizing", True):
            # Dynamic Risk Sizing (Kelly Criterion inspired)
            # We risk a percentage of the account proportional to AI confidence.
            # e.g. 100% confidence = 2% risk, 50% confidence = 1% risk.
            risk_per_trade_pct = (confidence / 100.0) * 0.02
            risk_per_trade_pct = min(risk_per_trade_pct, 0.02) # Cap maximum risk at 2% per trade
            
            dollar_risk = self.balance * risk_per_trade_pct
            size = dollar_risk / risk_distance
            trade_margin = size * self.current_price
            logging.info(f"Dynamic Sizing: Risking {risk_per_trade_pct*100:.2f}% (${dollar_risk:.2f}) based on {confidence}% confidence.")
        else:
            # Fixed Margin Fraction Sizing
            # Use the "Trade Margin (%)" setting from the UI, e.g. 20% of total balance
            fraction = margin_fraction if margin_fraction else state.settings.get("trade_margin", 0.20)
            trade_margin = self.balance * fraction
            size = trade_margin / self.current_price
            logging.info(f"Fixed Sizing: Using {fraction*100:.2f}% of balance as margin.")
        
        # Cap max position size to 50% of account to prevent over-exposure
        max_margin = self.balance * 0.50
        if trade_margin > max_margin:
            trade_margin = max_margin
            size = trade_margin / self.current_price
            logging.info("Capped position size to 50% of available balance.")
        
        if trade_margin > self.balance:
            logging.warning("Insufficient balance to open position.")
            return

        position = {
            "type": position_type,
            "entry_price": self.current_price,
            "size": size,
            "target": target,
            "stop_loss": stop_loss,
            "open_time": datetime.now(),
            "ai_reasoning": reasoning
        }
        self.positions.append(position)
        self.balance -= trade_margin
        
        self._save_active_positions()
        
        msg = f"🚀 **OPENED {position_type}** at {self.current_price:.2f}\n🎯 TP: {target}\n🛑 SL: {stop_loss}\n💰 Risk: ${trade_margin:.2f}\n🧠 AI Reason: {reasoning}"
        logging.info(msg.replace('**', '').replace('\n', ' | '))
        send_discord_alert(msg)

    def _close_position(self, pos, close_reason, exit_price):
        if pos["type"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else: # SHORT
            pnl = (pos["entry_price"] - exit_price) * pos["size"]
            
        trade_margin = pos["entry_price"] * pos["size"]
        self.balance += (trade_margin + pnl)
        
        if pos in self.positions:
            self.positions.remove(pos)
            self._save_active_positions()
        
        trade_record = {
            "type": pos["type"],
            "entry_price": pos["entry_price"],
            "close_price": exit_price,
            "target": pos["target"],
            "stop_loss": pos["stop_loss"],
            "pnl": pnl,
            "reason": close_reason,
            "margin_used": trade_margin,
            "ai_reasoning": pos.get("ai_reasoning", ""),
            "timestamp": pos.get("open_time", datetime.now()).isoformat() if not isinstance(pos.get("open_time", ""), str) else pos.get("open_time", datetime.now().isoformat())
        }
        
        db.save_trade(trade_record)
        self.trade_history.append(trade_record)
        
        emoji = "✅" if pnl > 0 else "❌"
        msg = f"{emoji} **CLOSED {pos['type']}** | Reason: {close_reason}\n💸 PNL: ${pnl:.2f}\n🏦 New Balance: ${self.balance:.2f}"
        logging.info(msg.replace('**', '').replace('\n', ' | '))
        send_discord_alert(msg)

    def _check_positions(self):
        positions_changed = False
        for pos in self.positions[:]:
            # Trailing Stop Loss Logic using ATR (e.g., 1.5 * ATR)
            # Fallback to 0.5% if ATR is not available
            trail_distance = (1.5 * self.current_atr) if self.current_atr > 0 else (self.current_price * 0.005)
            
            if pos["type"] == "LONG":
                # Initialize or update peak price
                if 'peak_price' not in pos or self.current_price > pos['peak_price']:
                    pos['peak_price'] = self.current_price
                    positions_changed = True
                    # If we are profitable, move the stop loss up
                    if pos['peak_price'] > pos['entry_price']:
                        new_sl = pos['peak_price'] - trail_distance
                        if new_sl > pos['stop_loss']:
                            pos['stop_loss'] = new_sl
                            logging.info(f"📈 Trailing SL moved up to {new_sl:.2f} for LONG position")

                if self.current_price >= pos["target"]:
                    self._close_position(pos, "TAKE_PROFIT", pos["target"])
                elif self.current_price <= pos["stop_loss"]:
                    self._close_position(pos, "STOP_LOSS/TRAILING", pos["stop_loss"])
            
            elif pos["type"] == "SHORT":
                # Initialize or update floor price
                if 'floor_price' not in pos or self.current_price < pos['floor_price']:
                    pos['floor_price'] = self.current_price
                    positions_changed = True
                    # If we are profitable, move the stop loss down
                    if pos['floor_price'] < pos['entry_price']:
                        new_sl = pos['floor_price'] + trail_distance
                        if new_sl < pos['stop_loss']:
                            pos['stop_loss'] = new_sl
                            logging.info(f"📉 Trailing SL moved down to {new_sl:.2f} for SHORT position")

                if self.current_price <= pos["target"]:
                    self._close_position(pos, "TAKE_PROFIT", pos["target"])
                elif self.current_price >= pos["stop_loss"]:
                    self._close_position(pos, "STOP_LOSS/TRAILING", pos["stop_loss"])
                    
        if positions_changed:
            self._save_active_positions()

    def get_recent_trades(self, limit=3):
        """Returns the most recent completed trades for the AI to learn from."""
        return self.trade_history[-limit:] if self.trade_history else []

    def get_portfolio_value(self):
        value = self.balance
        for pos in self.positions:
            unrealized_pnl = 0
            if pos["type"] == "LONG":
                unrealized_pnl = (self.current_price - pos["entry_price"]) * pos["size"]
            else:
                unrealized_pnl = (pos["entry_price"] - self.current_price) * pos["size"]
            trade_margin = pos["entry_price"] * pos["size"]
            value += (trade_margin + unrealized_pnl)
        return value

    def get_state(self):
        formatted_positions = []
        for pos in self.positions:
            unrealized_pnl = 0
            if pos["type"] == "LONG":
                unrealized_pnl = (self.current_price - pos["entry_price"]) * pos["size"]
            else:
                unrealized_pnl = (pos["entry_price"] - self.current_price) * pos["size"]
                
            formatted_positions.append({
                "type": pos["type"],
                "entry_price": pos["entry_price"],
                "size": pos["size"],
                "target": pos["target"],
                "stop_loss": pos["stop_loss"],
                "open_time": pos["open_time"].isoformat() if hasattr(pos["open_time"], 'isoformat') else str(pos["open_time"]),
                "unrealized_pnl": unrealized_pnl
            })
            
        return {
            "balance": self.balance,
            "portfolio_value": self.get_portfolio_value(),
            "positions": formatted_positions,
            "history": self.trade_history
        }
        
    def print_status(self):
        logging.info(f"📊 --- PAPER TRADING STATUS ---")
        logging.info(f"💰 Total Portfolio Value: ${self.get_portfolio_value():.2f}")
        logging.info(f"💵 Available Balance: ${self.balance:.2f}")
        if self.positions:
            logging.info(f"📈 Open Positions: {len(self.positions)}")
        else:
            logging.info(f"📉 No open positions.")
        logging.info(f"--------------------------------")
