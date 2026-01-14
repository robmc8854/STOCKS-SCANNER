"""
TELEGRAM BOT - FIXED VERSION
Handles alerts and messages with proper error handling
"""

import requests
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for sending trading alerts
    """
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = True
        
        # Test connection
        if not self.test_connection():
            logger.error("❌ Telegram connection failed!")
            self.enabled = False
        else:
            logger.info("✅ Telegram connected successfully")
    
    def test_connection(self) -> bool:
        """Test if bot token and chat ID are valid"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    logger.info(f"✅ Bot connected: @{data['result'].get('username')}")
                    return True
            
            logger.error(f"❌ Telegram API error: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Telegram connection error: {e}")
            return False
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram
        
        Args:
            message: Message text (supports HTML)
            parse_mode: "HTML" or "Markdown"
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("⚠️ Telegram disabled - message not sent")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    logger.info(f"✅ Message sent to Telegram")
                    return True
                else:
                    logger.error(f"❌ Telegram API error: {data.get('description')}")
                    return False
            else:
                logger.error(f"❌ HTTP error {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Telegram request timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    def send_setup_alert(self, setup: dict) -> bool:
        """Send formatted setup alert"""
        try:
            risk = setup['entry'] - setup['stop']
            risk_pct = (risk / setup['entry']) * 100
            
            message = f"""
🚀 <b>TRADE SIGNAL - {setup['ticker']}</b>

✅ <b>Status: VALIDATED SETUP</b>

📊 <b>Setup:</b> {setup.get('setup_type', 'BREAKOUT')}
⭐ <b>Score:</b> {setup['score']}/100

💰 <b>ENTRY:</b> ${setup['entry']:.2f}
🛑 <b>STOP:</b> ${setup['stop']:.2f} ({risk_pct:.1f}%)
🎯 <b>TARGET 1R:</b> ${setup['target_1R']:.2f} ({risk_pct:.1f}%)
🎯 <b>TARGET 2R:</b> ${setup['target_2R']:.2f} ({risk_pct*2:.1f}%)

📈 <b>R:R Ratio:</b> 1:2

<b>📋 ENHANCED METRICS:</b>
"""
            
            # Add Phase 2 metrics if available
            if 'clenow_momentum' in setup:
                message += f"• Clenow Momentum: {setup['clenow_momentum']:.3f}\n"
            
            if 'rs_vs_spy' in setup:
                message += f"• RS vs SPY: {setup['rs_vs_spy']:.2f}x\n"
            
            if 'yz_volatility' in setup:
                message += f"• YZ Volatility: {setup['yz_volatility']:.1f}%\n"
            
            # Add Phase 3 metrics if available
            if 'trend_persistence' in setup:
                trend = setup['trend_persistence']
                trend_str = "Trending" if trend > 0 else "Mean-reverting"
                message += f"• Trend: {trend_str} ({trend:.2f})\n"
            
            if 'sector' in setup:
                message += f"• Sector: {setup['sector']}\n"
            
            message += f"\n<b>Notes:</b> {setup.get('notes', 'Top momentum setup')}"
            
            message += f"\n\n✅ <b>ACTION: PLACE TRADE</b>"
            message += f"\n📱 <b>Follow Qullamaggie rules:</b>"
            message += f"\n   • Enter at ${setup['entry']:.2f}"
            message += f"\n   • Stop at ${setup['stop']:.2f}"
            message += f"\n   • Scale out: 1/3 at 1R, 1/3 at 2R"
            message += f"\n   • Trail stop with 10MA"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error formatting alert: {e}")
            return False
    
    def send_scan_summary(self, scan_results: dict) -> bool:
        """Send scan summary"""
        try:
            message = f"""
📊 <b>SCAN COMPLETE - {datetime.now().strftime('%H:%M')}</b>

🔍 Scanned: {scan_results.get('total_scanned', 0)} stocks
✅ Found: {scan_results.get('setups_found', 0)} setups

"""
            
            if scan_results.get('setups_found', 0) > 0:
                message += f"⭐ Top Score: {scan_results.get('top_score', 0)}/100\n"
                message += f"📈 Avg Score: {scan_results.get('avg_score', 0):.1f}/100\n"
            else:
                message += "No setups met strict Qullamaggie criteria today\n"
            
            message += f"\n⏰ Next scan: {scan_results.get('next_scan', 'N/A')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending summary: {e}")
            return False
    
    def send_position_alert(self, position: dict, alert_type: str) -> bool:
        """Send position management alert"""
        try:
            if alert_type == 'partial_profit':
                message = f"""
💰 <b>TAKE PARTIAL PROFITS - {position['ticker']}</b>

📊 Entry: ${position['entry_price']:.2f}
💵 Current: ${position['current_price']:.2f}
📈 P&L: {position['pnl_pct']:.1f}%

⚠️ <b>ACTION: SELL 1/3 TO 1/2</b>
🛑 Move stop to breakeven

Days held: {position.get('days_held', 0)}
"""
            
            elif alert_type == 'trail_stop':
                message = f"""
🛑 <b>EXIT SIGNAL - {position['ticker']}</b>

📊 Entry: ${position['entry_price']:.2f}
💵 Current: ${position['current_price']:.2f}
📉 Close below 10MA

⚠️ <b>ACTION: EXIT POSITION</b>

P&L: {position['pnl_pct']:.1f}%
Days held: {position.get('days_held', 0)}
"""
            
            else:
                return False
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending position alert: {e}")
            return False
    
    def send_error(self, error_message: str) -> bool:
        """Send error notification"""
        try:
            message = f"""
⚠️ <b>SYSTEM ERROR</b>

{error_message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            return self.send_message(message)
        except:
            return False


def create_telegram_bot(token: str, chat_id: str) -> Optional[TelegramBot]:
    """
    Create and test Telegram bot
    
    Returns:
        TelegramBot instance or None if failed
    """
    try:
        bot = TelegramBot(token, chat_id)
        if bot.enabled:
            return bot
        else:
            logger.error("Telegram bot disabled due to connection failure")
            return None
    except Exception as e:
        logger.error(f"Failed to create Telegram bot: {e}")
        return None


# Convenience function for dashboard
def send_telegram_alert(bot: Optional[TelegramBot], message: str) -> bool:
    """
    Send message via telegram bot if available
    
    Args:
        bot: TelegramBot instance or None
        message: Message to send
    
    Returns:
        True if sent, False otherwise
    """
    if bot is None:
        logger.warning("⚠️ Telegram bot not available")
        return False
    
    return bot.send_message(message)
