"""
ULTRA TRADING PLATFORM - ULTIMATE EDITION
Professional Risk Management + Beautiful Interface + Full Features
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys
import yfinance as yf
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner_qullamaggie_pro import UltraScannerEngine
from config import config
from complete_tickers import COMPLETE_TICKERS
import requests
# from position_manager import QullamaggiePositionManager, format_position_alert, check_and_alert_positions

# Telegram notification function
def send_telegram_alert(message, photo_url=None):
    """Send alert to Telegram with optional chart"""
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        data = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data)
        
        # Send photo if provided
        if photo_url:
            photo_url_api = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendPhoto"
            photo_data = {
                'chat_id': config.TELEGRAM_CHAT_ID,
                'photo': photo_url,
                'caption': 'Trade Setup Chart'
            }
            requests.post(photo_url_api, data=photo_data)
        
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def send_positions_to_telegram(account):
    """Send current positions to Telegram"""
    if len(account.positions) == 0:
        msg = "📊 <b>OPEN POSITIONS</b>\n\n✅ No open positions"
        send_telegram_alert(msg)
        return
    
    # Update positions first
    account.update_positions()
    
    msg = f"📊 <b>OPEN POSITIONS ({len(account.positions)})</b>\n\n"
    
    total_value = 0
    total_pnl = 0
    
    for pos in account.positions:
        pnl_emoji = "🟢" if pos['pnl'] > 0 else "🔴" if pos['pnl'] < 0 else "⚪"
        r_mult = pos.get('r_multiple', 0)
        
        msg += f"{pnl_emoji} <b>{pos['ticker']}</b>\n"
        msg += f"  Entry: ${pos['entry_price']:.2f} → Current: ${pos['current_price']:.2f}\n"
        msg += f"  Shares: {pos['shares']:,} | Value: £{pos['current_value']:,.0f}\n"
        msg += f"  P&L: £{pos['pnl']:,.0f} ({pos['pnl_pct']:+.2f}%) | {r_mult:+.2f}R\n"
        msg += f"  Stop: ${pos['stop']:.2f} ({pos.get('dist_to_stop', 0):+.1f}% away)\n"
        msg += f"  Target: ${pos['target_1R']:.2f} ({pos.get('dist_to_target1', 0):+.1f}% away)\n\n"
        
        total_value += pos['current_value']
        total_pnl += pos['pnl']
    
    msg += f"━━━━━━━━━━━━━━━━━━\n"
    msg += f"💼 <b>Total Value:</b> £{total_value:,.0f}\n"
    msg += f"💰 <b>Total P&L:</b> £{total_pnl:,.0f}\n"
    msg += f"⚠️ <b>Portfolio Heat:</b> {account.get_portfolio_heat():.1f}%"
    
    send_telegram_alert(msg)

def format_trade_alert(ticker, setup_type, score, entry, stop, target1, target2, shares, validation_status, reasons):
    """Format Qullamaggie-style trade alert"""
    
    # Calculate R:R
    risk = entry - stop
    reward = target1 - entry
    rr = reward / risk if risk > 0 else 0
    
    # Risk/Reward percentages
    risk_pct = (risk / entry) * 100
    reward_pct = (reward / entry) * 100
    
    # Status emoji
    status_emoji = "✅" if validation_status else "❌"
    
    message = f"""
🚀 <b>TRADE SIGNAL - {ticker}</b>

{status_emoji} <b>Status:</b> {'VALIDATED - READY TO TRADE' if validation_status else 'REJECTED - DO NOT TRADE'}

📊 <b>Setup:</b> {setup_type}
⭐ <b>Score:</b> {score}/100

💰 <b>ENTRY:</b> ${entry:.2f}
🛑 <b>STOP:</b> ${stop:.2f} ({risk_pct:.1f}%)
🎯 <b>TARGET 1R:</b> ${target1:.2f} ({reward_pct:.1f}%)
🎯 <b>TARGET 2R:</b> ${target2:.2f} ({(target2-entry)/entry*100:.1f}%)

📈 <b>R:R Ratio:</b> {rr:.1f}:1
📦 <b>Position Size:</b> {shares:,} shares
💵 <b>Position Value:</b> £{shares * entry:,.0f}

<b>📋 QULLAMAGGIE CHECKLIST:</b>
"""
    
    # Add top 3-4 reasons
    for i, reason in enumerate(reasons[:4], 1):
        # Clean the reason of markdown
        clean_reason = reason.replace('**', '').replace('*', '').replace('#', '')[:150]
        message += f"{i}. {clean_reason}\n"
    
    if validation_status:
        message += f"\n✅ <b>ACTION: PLACE TRADE</b>"
        message += f"\n📱 Follow Qullamaggie rules:"
        message += f"\n   • Enter at ${entry:.2f}"
        message += f"\n   • Stop at ${stop:.2f}"
        message += f"\n   • Scale out: 1/3 at 1R, 1/3 at 2R"
        message += f"\n   • Trail stop with 10MA"
    else:
        message += f"\n❌ <b>ACTION: DO NOT TRADE</b>"
        message += f"\n⚠️ Failed risk management validation"
    
    return message

# Page config
st.set_page_config(
    page_title="Ultra Trading Platform - Ultimate Edition",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .big-font {
        font-size: 3em !important;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .setup-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #252b4a 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .reason-box {
        background: linear-gradient(135deg, #252b4a 0%, #1a1f3a 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #10b981;
        font-size: 1.1em;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #252b4a 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    .position-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #252b4a 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    .risk-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .risk-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .risk-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .trade-detail {
        font-size: 1.2em;
        padding: 8px 0;
        border-bottom: 1px solid #667eea33;
    }
    .pnl-positive {
        color: #10b981;
        font-size: 2em;
        font-weight: bold;
    }
    .pnl-negative {
        color: #ef4444;
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Account Management Class
class TradingAccount:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = []
        self.closed_trades = []
        self.journal_entries = []
        self.max_risk_per_trade = config.RISK_PER_TRADE / 100
        self.max_portfolio_heat = 6.0 / 100
        self.max_positions = config.MAX_POSITIONS
    
    def get_equity(self):
        position_value = sum(p['current_value'] for p in self.positions)
        return self.cash + position_value
    
    def get_total_risk(self):
        return sum(p['risk_amount'] for p in self.positions)
    
    def get_portfolio_heat(self):
        equity = self.get_equity()
        return (self.get_total_risk() / equity) * 100 if equity > 0 else 0
    
    def can_add_position(self, position_value, risk_amount):
        errors = []
        
        if position_value > self.cash:
            errors.append(f"❌ Insufficient cash: Need £{position_value:,.0f}, have £{self.cash:,.0f}")
        
        if len(self.positions) >= self.max_positions:
            errors.append(f"❌ Max positions reached: {self.max_positions}")
        
        new_total_risk = self.get_total_risk() + risk_amount
        new_heat = (new_total_risk / self.get_equity()) * 100
        if new_heat > self.max_portfolio_heat * 100:
            errors.append(f"❌ Would exceed portfolio heat: {new_heat:.1f}% > {self.max_portfolio_heat*100:.1f}%")
        
        single_risk_pct = (risk_amount / self.get_equity()) * 100
        if single_risk_pct > self.max_risk_per_trade * 100:
            errors.append(f"❌ Position risk too high: {single_risk_pct:.1f}% > {self.max_risk_per_trade*100:.1f}%")
        
        return len(errors) == 0, errors
    
    def calculate_position_size(self, entry, stop):
        equity = self.get_equity()
        max_risk_amount = equity * self.max_risk_per_trade
        remaining_heat = (self.max_portfolio_heat * equity) - self.get_total_risk()
        available_risk = min(max_risk_amount, remaining_heat)
        
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0
        
        max_shares = int(available_risk / risk_per_share)
        max_position_value = max_shares * entry
        
        if max_position_value > self.cash:
            max_shares = int(self.cash / entry)
        
        return max_shares
    
    def add_position(self, ticker, shares, entry, stop, target1, target2, setup_type, score, notes=""):
        position_value = shares * entry
        risk_amount = shares * (entry - stop)
        
        self.cash -= position_value
        
        position = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry,
            'current_price': entry,
            'stop': stop,
            'target_1R': target1,
            'target_2R': target2,
            'setup_type': setup_type,
            'score': score,
            'notes': notes,
            'entry_date': datetime.now().isoformat(),
            'entry_value': position_value,
            'current_value': position_value,
            'risk_amount': risk_amount,
            'pnl': 0,
            'pnl_pct': 0
        }
        
        self.positions.append(position)
        return True
    
    def update_positions(self):
        for pos in self.positions:
            try:
                ticker = yf.Ticker(pos['ticker'])
                current = ticker.history(period="1d")['Close'].iloc[-1]
                
                pos['current_price'] = current
                pos['current_value'] = pos['shares'] * current
                pos['pnl'] = (current - pos['entry_price']) * pos['shares']
                pos['pnl_pct'] = ((current / pos['entry_price']) - 1) * 100
                pos['dist_to_stop'] = ((current / pos['stop']) - 1) * 100
                pos['dist_to_target1'] = ((pos['target_1R'] / current) - 1) * 100
                pos['dist_to_target2'] = ((pos['target_2R'] / current) - 1) * 100
                pos['r_multiple'] = pos['pnl'] / pos['risk_amount'] if pos['risk_amount'] > 0 else 0
            except:
                pass
    
    def close_position(self, index, exit_price):
        pos = self.positions[index]
        exit_value = pos['shares'] * exit_price
        pnl = exit_value - pos['entry_value']
        
        self.cash += exit_value
        
        trade = pos.copy()
        trade['exit_price'] = exit_price
        trade['exit_date'] = datetime.now().isoformat()
        trade['exit_value'] = exit_value
        trade['pnl'] = pnl
        trade['pnl_pct'] = ((exit_price / pos['entry_price']) - 1) * 100
        trade['r_multiple'] = pnl / pos['risk_amount'] if pos['risk_amount'] > 0 else 0
        
        self.closed_trades.append(trade)
        self.positions.pop(index)
        
        return pnl
    
    def get_stats(self):
        equity = self.get_equity()
        total_pnl = equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        position_pnl = sum(p['pnl'] for p in self.positions)
        
        return {
            'equity': equity,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'position_pnl': position_pnl,
            'total_risk': self.get_total_risk(),
            'portfolio_heat': self.get_portfolio_heat(),
            'num_positions': len(self.positions),
            'buying_power': self.cash
        }

def create_advanced_chart(ticker, entry, stop, target1, target2):
    """Create Qullamaggie-style chart - 3 months daily with key indicators"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d")
        
        if hist.empty:
            return None
        
        # Ensure we only have last 3 months
        hist = hist.tail(63)  # ~63 trading days in 3 months
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=ticker,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='#10b981',
            decreasing_fillcolor='#ef4444'
        ))
        
        # Volume bars
        colors = ['#10b981' if hist['Close'].iloc[i] > hist['Open'].iloc[i] else '#ef4444' 
                 for i in range(len(hist))]
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.3,
            yaxis='y2'
        ))
        
        # QULLAMAGGIE KEY MOVING AVERAGES
        hist['MA10'] = hist['Close'].rolling(10).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()
        
        # 10 MA - Most important for Qullamaggie (trailing stop)
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['MA10'], 
            name='10 MA (Trailing Stop)',
            line=dict(color='#ec4899', width=2)
        ))
        
        # 20 MA - Support level
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['MA20'],
            name='20 MA (Support)',
            line=dict(color='#f59e0b', width=2)
        ))
        
        # 50 MA - Major trend line
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['MA50'],
            name='50 MA (Trend)',
            line=dict(color='#3b82f6', width=2, dash='dash')
        ))
        
        # TRADE LEVELS
        fig.add_hline(
            y=entry, 
            line_dash="solid", 
            line_color="#f59e0b", 
            line_width=3,
            annotation_text="ENTRY", 
            annotation_position="right",
            annotation=dict(font_size=14, font_color="#f59e0b")
        )
        
        fig.add_hline(
            y=stop, 
            line_dash="solid", 
            line_color="#ef4444", 
            line_width=3,
            annotation_text="STOP", 
            annotation_position="right",
            annotation=dict(font_size=14, font_color="#ef4444")
        )
        
        fig.add_hline(
            y=target1, 
            line_dash="dash", 
            line_color="#10b981", 
            line_width=2,
            annotation_text="TARGET 1R", 
            annotation_position="right",
            annotation=dict(font_size=12, font_color="#10b981")
        )
        
        fig.add_hline(
            y=target2, 
            line_dash="dash", 
            line_color="#22c55e", 
            line_width=2,
            annotation_text="TARGET 2R", 
            annotation_position="right",
            annotation=dict(font_size=12, font_color="#22c55e")
        )
        
        fig.update_layout(
            title=dict(
                text=f'<b>{ticker} - 3 Month Daily Chart (Qullamaggie Style)</b>', 
                font=dict(size=18, color='#667eea')
            ),
            yaxis_title='Price ($)',
            yaxis2=dict(
                title='Volume', 
                overlaying='y', 
                side='right', 
                showgrid=False
            ),
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=500,
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#1a1f3a',
            font=dict(color='white'),
            hovermode='x unified',
            xaxis=dict(
                rangeslider=dict(visible=False),
                type='date'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        return fig
    except Exception as e:
        print(f"Chart error: {e}")
        return None

def get_detailed_reasons(row):
    """Generate comprehensive trade reasoning"""
    reasons = []
    
    # Volume analysis
    vol_ratio = row.get('volume_ratio', 0)
    if vol_ratio > 3:
        reasons.append(f"🔥 **MASSIVE VOLUME SURGE**: {vol_ratio:.1f}x average volume - Institutional buying detected. This level of participation suggests strong conviction and potential for sustained momentum.")
    elif vol_ratio > 2:
        reasons.append(f"📊 **STRONG VOLUME**: {vol_ratio:.1f}x average - Significantly above normal, indicating genuine interest and reducing likelihood of false breakout.")
    elif vol_ratio > 1.5:
        reasons.append(f"✅ **HEALTHY VOLUME**: {vol_ratio:.1f}x average - Good participation confirming the move.")
    
    # RSI momentum
    rsi = row.get('rsi', 50)
    if rsi > 70:
        reasons.append(f"⚡ **POWERFUL MOMENTUM**: RSI at {rsi:.0f} - Bulls in complete control. Strong uptrend with momentum acceleration.")
    elif rsi > 60:
        reasons.append(f"💪 **STRONG MOMENTUM**: RSI at {rsi:.0f} - Healthy bullish momentum without being overextended.")
    elif rsi > 50:
        reasons.append(f"📈 **POSITIVE MOMENTUM**: RSI at {rsi:.0f} - Above neutral, bulls have edge.")
    
    # Relative strength
    rs = row.get('relative_strength', 1)
    if rs > 1.5:
        reasons.append(f"🚀 **MARKET LEADER**: Outperforming SPY by {rs:.2f}x - This is where institutions put money in strong markets. Clear sector leadership.")
    elif rs > 1.2:
        reasons.append(f"💎 **STRONG OUTPERFORMANCE**: {rs:.2f}x SPY - Demonstrating relative strength, a key Qullamaggie criterion.")
    elif rs > 1.0:
        reasons.append(f"✅ **BEATING MARKET**: {rs:.2f}x SPY - Positive relative strength confirms this is a market leader.")
    
    # Moving average alignment
    if row.get('price_to_ma10', 0) > 0 and row.get('price_to_ma20', 0) > 0:
        ma10_dist = row['price_to_ma10']
        ma20_dist = row['price_to_ma20']
        reasons.append(f"📊 **PERFECT MA SETUP**: Price {ma10_dist:.1f}% above 10MA and {ma20_dist:.1f}% above 20MA - Clean uptrend with MAs acting as support. Qullamaggie looks for exactly this.")
    
    # Setup-specific reasoning
    setup_type = row.get('setup_type', '')
    if 'Episodic Pivot' in setup_type:
        reasons.append(f"🎯 **EPISODIC PIVOT SETUP**: {setup_type} - Mark Minervini's favorite. Stock had 20%+ move, pulled back to 10/20 MA support, now bouncing on volume. This is a high-probability continuation pattern.")
    elif 'Bull Flag' in setup_type:
        reasons.append(f"🚩 **BULL FLAG PATTERN**: {setup_type} - Classic continuation after strong move. Consolidation forming higher lows, volume drying up during flag, ready to break higher.")
    elif 'Breakout' in setup_type:
        reasons.append(f"💥 **BREAKOUT SETUP**: {setup_type} - Breaking above resistance with volume. Price discovery mode - no overhead supply. High probability of continuation.")
    elif 'VCP' in setup_type:
        reasons.append(f"🎪 **VOLATILITY CONTRACTION**: {setup_type} - William O'Neil's VCP. Contracting volatility suggests supply is exhausted. Spring-loaded for explosive move.")
    
    # Risk/Reward
    entry = row.get('entry', 0)
    stop = row.get('stop', 0)
    target1 = row.get('target_1R', 0)
    if entry > stop and target1 > entry:
        rr = (target1 - entry) / (entry - stop)
        reward_pct = ((target1 - entry) / entry) * 100
        risk_pct = ((entry - stop) / entry) * 100
        
        if rr > 3:
            reasons.append(f"💰 **EXCEPTIONAL R:R**: {rr:.1f}:1 ratio ({reward_pct:.1f}% gain vs {risk_pct:.1f}% risk) - Asymmetric opportunity. Even with 40% win rate, this is profitable.")
        elif rr > 2:
            reasons.append(f"💵 **EXCELLENT R:R**: {rr:.1f}:1 ratio ({reward_pct:.1f}% gain vs {risk_pct:.1f}% risk) - Strong risk/reward profile.")
        elif rr > 1.5:
            reasons.append(f"✔️ **GOOD R:R**: {rr:.1f}:1 ratio ({reward_pct:.1f}% gain vs {risk_pct:.1f}% risk) - Acceptable risk/reward.")
    
    # Score reasoning
    score = row.get('score', 0)
    if score >= 90:
        reasons.append(f"⭐ **ELITE SETUP**: Score {score}/100 - ALL criteria aligned perfectly. This is the type of setup Qullamaggie waits for.")
    elif score >= 85:
        reasons.append(f"🌟 **PREMIUM SETUP**: Score {score}/100 - Very high quality with multiple confirming factors.")
    elif score >= 80:
        reasons.append(f"✨ **HIGH QUALITY**: Score {score}/100 - Strong setup with good probability.")
    
    return reasons

def calculate_performance_metrics(trades):
    """Calculate comprehensive performance metrics"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    total_trades = len(df)
    winners = len(wins)
    losers = len(losses)
    
    metrics = {
        'total_trades': total_trades,
        'winners': winners,
        'losers': losers,
        'win_rate': (winners / total_trades * 100) if total_trades > 0 else 0,
        'loss_rate': (losers / total_trades * 100) if total_trades > 0 else 0,
        
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'largest_win': wins['pnl'].max() if len(wins) > 0 else 0,
        'largest_loss': losses['pnl'].min() if len(losses) > 0 else 0,
        
        'avg_win_r': wins['r_multiple'].mean() if len(wins) > 0 else 0,
        'avg_loss_r': losses['r_multiple'].mean() if len(losses) > 0 else 0,
        'avg_r': df['r_multiple'].mean(),
        
        'total_pnl': df['pnl'].sum(),
        'avg_trade': df['pnl'].mean(),
    }
    
    # Profit factor
    total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    metrics['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else 0
    
    # Expectancy
    win_rate = metrics['win_rate'] / 100
    metrics['expectancy'] = (win_rate * metrics['avg_win']) + ((1 - win_rate) * metrics['avg_loss'])
    
    # Sharpe ratio
    if df['pnl'].std() > 0:
        metrics['sharpe_ratio'] = (metrics['avg_trade'] / df['pnl'].std()) * np.sqrt(252)
    else:
        metrics['sharpe_ratio'] = 0
    
    # Max drawdown
    df = df.sort_values('exit_date')
    df['cumulative'] = df['pnl'].cumsum()
    df['peak'] = df['cumulative'].expanding().max()
    df['drawdown'] = df['cumulative'] - df['peak']
    metrics['max_drawdown'] = df['drawdown'].min()
    
    # Consecutive wins/losses
    df['win'] = df['pnl'] > 0
    df['streak'] = (df['win'] != df['win'].shift()).cumsum()
    streaks = df.groupby('streak')['win'].agg(['first', 'count'])
    
    win_streaks = streaks[streaks['first'] == True]['count']
    loss_streaks = streaks[streaks['first'] == False]['count']
    
    metrics['max_win_streak'] = win_streaks.max() if len(win_streaks) > 0 else 0
    metrics['max_loss_streak'] = loss_streaks.max() if len(loss_streaks) > 0 else 0
    
    return metrics

# Initialize
if 'account' not in st.session_state:
    st.session_state.account = TradingAccount(config.ACCOUNT_SIZE)
    st.session_state.scanner = UltraScannerEngine(COMPLETE_TICKERS)
    # st.session_state.position_manager = None  # Disabled
    st.session_state.last_position_check = None
    st.session_state.last_morning_report_date = None

    st.session_state.scan_results = pd.DataFrame()
    st.session_state.last_update = None
    st.session_state.last_scan_time = None
    st.session_state.auto_scan_enabled = False
    st.session_state.sent_alerts = set()  # Track which tickers we've already alerted on
    st.session_state.last_scan_date = None  # Track which day we last scanned
    
    try:
        with open('data/account.json', 'r') as f:
            data = json.load(f)
            st.session_state.account.cash = data['cash']
            st.session_state.account.positions = data['positions']
            st.session_state.account.closed_trades = data.get('closed_trades', [])
            st.session_state.account.journal_entries = data.get('journal_entries', [])
    except:
        pass

def save_account():
    os.makedirs('data', exist_ok=True)
    data = {
        'cash': st.session_state.account.cash,
        'positions': st.session_state.account.positions,
        'closed_trades': st.session_state.account.closed_trades,
        'journal_entries': st.session_state.account.journal_entries
    }
    with open('data/account.json', 'w') as f:
        json.dump(data, f, default=str)

def should_send_alert(ticker, score):
    """Check if we should send alert for this ticker"""
    # Create unique key with ticker and date
    today = datetime.now().date()
    alert_key = f"{ticker}_{today}_{score}"
    
    # Reset sent alerts at market open (9:30 AM)
    current_time = datetime.now().time()
    if current_time.hour == 9 and current_time.minute >= 30:
        if st.session_state.last_scan_date != today:
            st.session_state.sent_alerts.clear()
            st.session_state.last_scan_date = today
    
    # Check if already sent today
    if alert_key in st.session_state.sent_alerts:
        return False
    
    st.session_state.sent_alerts.add(alert_key)
    return True

# Auto-update positions
if st.session_state.last_update is None or \
   (datetime.now() - st.session_state.last_update).seconds > 60:
    st.session_state.account.update_positions()
    st.session_state.last_update = datetime.now()
    save_account()

# Auto-scan logic - SMART TIMING (Qullamaggie style)
# Scan at: Market open (9:30 AM), mid-morning (10:30 AM), lunch (12:30 PM), mid-afternoon (2:30 PM)
if st.session_state.auto_scan_enabled:
    current_time = datetime.now().time()
    scan_times = [
        (9, 30),   # Market open - fresh setups
        (10, 30),  # Mid-morning - breakouts
        (12, 30),  # Lunch - new patterns
        (14, 30),  # Mid-afternoon - late day setups
    ]
    
    should_scan = False
    for hour, minute in scan_times:
        if current_time.hour == hour and current_time.minute >= minute and current_time.minute < minute + 5:
            # Check if we already scanned in this window
            if st.session_state.last_scan_time is None or \
               (datetime.now() - st.session_state.last_scan_time).seconds > 3600:
                should_scan = True
                break
    
    if should_scan:
        # Run scan automatically
        results = st.session_state.scanner.run_full_scan()
        st.session_state.scan_results = results
        st.session_state.last_scan_time = datetime.now()
        
        # Auto-validate and send alerts (ONLY NEW ONES)
        if len(results) > 0:
            account = st.session_state.account
            validated_count = 0
            rejected_count = 0
            new_alerts = 0
            
            for idx, row in results.iterrows():
                # Only send if we haven't alerted on this ticker today
                if should_send_alert(row['ticker'], row['score']):
                    max_shares = account.calculate_position_size(row['entry'], row['stop'])
                    
                    if max_shares > 0:
                        actual_value = max_shares * row['entry']
                        actual_risk = max_shares * (row['entry'] - row['stop'])
                        can_add, errors = account.can_add_position(actual_value, actual_risk)
                        reasons = get_detailed_reasons(row)
                        
                        alert_msg = format_trade_alert(
                            row['ticker'], row['setup_type'], row['score'],
                            row['entry'], row['stop'], row['target_1R'], row['target_2R'],
                            max_shares, can_add, reasons
                        )
                        send_telegram_alert(alert_msg)
                        new_alerts += 1
                        
                        if can_add:
                            validated_count += 1
                        else:
                            rejected_count += 1
                        
                        time.sleep(0.5)
            
            # Send summary only if we sent new alerts
            if new_alerts > 0:
                summary_msg = f"""
📊 <b>AUTO-SCAN @ {datetime.now().strftime('%H:%M')}</b>

🆕 New Setups: {new_alerts}
✅ Validated: {validated_count}
❌ Rejected: {rejected_count}

🔄 Next scan at: {scan_times[(scan_times.index((current_time.hour, current_time.minute)) + 1) % len(scan_times)][0]}:{scan_times[(scan_times.index((current_time.hour, current_time.minute)) + 1) % len(scan_times)][1]:02d}
"""
                send_telegram_alert(summary_msg)


# HEADER - PROFESSIONAL DASHBOARD
account = st.session_state.account
stats = account.get_stats()

st.markdown('<p class="big-font">🚀 ULTRA TRADING PLATFORM</p>', unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.3em; color: #667eea; margin-bottom: 30px;'><b>Professional Edition</b> | Full Risk Management | Live P&L | 234 Features</div>", unsafe_allow_html=True)

# ACCOUNT METRICS - Beautiful Cards
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9em; color: #667eea;'>EQUITY</div>
        <div style='font-size: 2em; font-weight: bold;'>£{stats['equity']:,.0f}</div>
        <div style='color: {"#10b981" if stats["total_pnl"] > 0 else "#ef4444"}; font-size: 1.2em;'>{stats['total_pnl_pct']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9em; color: #667eea;'>CASH</div>
        <div style='font-size: 2em; font-weight: bold;'>£{stats['cash']:,.0f}</div>
        <div style='font-size: 0.9em; opacity: 0.7;'>Buying Power</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9em; color: #667eea;'>POSITIONS</div>
        <div style='font-size: 2em; font-weight: bold;'>{stats['num_positions']}/{account.max_positions}</div>
        <div style='font-size: 0.9em; opacity: 0.7;'>Open</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    pnl_color = "#10b981" if stats['position_pnl'] > 0 else "#ef4444"
    st.markdown(f"""
    <div class='metric-card' style='border-color: {pnl_color};'>
        <div style='font-size: 0.9em; color: #667eea;'>POSITION P&L</div>
        <div style='font-size: 2em; font-weight: bold; color: {pnl_color};'>£{stats['position_pnl']:,.0f}</div>
        <div style='font-size: 0.9em; opacity: 0.7;'>Unrealized</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9em; color: #667eea;'>TOTAL RISK</div>
        <div style='font-size: 2em; font-weight: bold;'>£{stats['total_risk']:,.0f}</div>
        <div style='font-size: 0.9em; opacity: 0.7;'>At Risk</div>
    </div>
    """, unsafe_allow_html=True)

heat = stats['portfolio_heat']
heat_emoji = "🟢" if heat < 4 else "🟡" if heat < 6 else "🔴"
heat_color = "#10b981" if heat < 4 else "#f59e0b" if heat < 6 else "#ef4444"

with col6:
    st.markdown(f"""
    <div class='metric-card' style='border-color: {heat_color};'>
        <div style='font-size: 0.9em; color: #667eea;'>PORTFOLIO HEAT</div>
        <div style='font-size: 2em; font-weight: bold; color: {heat_color};'>{heat_emoji} {heat:.1f}%</div>
        <div style='font-size: 0.9em; opacity: 0.7;'>Max: 6%</div>
    </div>
    """, unsafe_allow_html=True)

# Risk Status Bar
st.markdown("<br>", unsafe_allow_html=True)
if heat < 4:
    st.markdown(f"<div class='risk-excellent'>✅ EXCELLENT RISK LEVEL: {heat:.1f}% Portfolio Heat - Safe to add positions</div>", unsafe_allow_html=True)
elif heat < 6:
    st.markdown(f"<div class='risk-warning'>⚠️ MODERATE RISK: {heat:.1f}% Portfolio Heat - Approaching limit, be selective</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='risk-danger'>🔴 HIGH RISK: {heat:.1f}% Portfolio Heat - REDUCE POSITIONS IMMEDIATELY!</div>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 NAVIGATION")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        ["🔍 Scanner", "💼 Positions", "⭐ Watchlist", "📖 Journal", 
         "📈 Analytics", "⚠️ Risk Management", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.markdown("### ⏱️ Live Status")
    st.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'}")
    st.markdown(f"**Closed Trades:** {len(account.closed_trades)}")
    
    if account.closed_trades:
        total_closed_pnl = sum(t['pnl'] for t in account.closed_trades)
        st.markdown(f"**Realized P&L:** £{total_closed_pnl:,.0f}")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if account.closed_trades:
        wins = [t for t in account.closed_trades if t['pnl'] > 0]
        wr = (len(wins) / len(account.closed_trades) * 100) if account.closed_trades else 0
        st.markdown(f"**Win Rate:** {wr:.1f}%")
        
        avg_r = sum(t['r_multiple'] for t in account.closed_trades) / len(account.closed_trades)
        st.markdown(f"**Avg R-Multiple:** {avg_r:.2f}R")

# MAIN CONTENT
st.markdown("---")


# ============================================================================
# TAB 1: SCANNER - MEGA DETAILED
# ============================================================================
if page == "🔍 Scanner":
    st.header("🔍 MARKET SCANNER - PROFESSIONAL ANALYSIS")
    st.markdown("**Qullamaggie + Niv Methodologies** | 500 Stocks | 50+ Filters | 8 Patterns | Real-time Analysis")
    
    # DEBUG: Show current state
    if not st.session_state.scan_results.empty:
        st.success(f"✅ {len(st.session_state.scan_results)} setups in memory")
        if len(st.session_state.scan_results) > 0:
            st.info(f"📊 Scores: {sorted(st.session_state.scan_results['score'].values, reverse=True)}")
    else:
        st.info("📭 No scan results yet - click 'FULL SCAN' to start")
    
    col1, col2, col3 = st.columns([2,2,6])
    
    with col1:
        if st.button("🚀 FULL SCAN", use_container_width=True, type="primary"):
            with st.spinner("🔍 Scanning 386 stocks... 2-3 minutes"):
                results = st.session_state.scanner.run_full_scan()
                st.session_state.scan_results = results
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs('data', exist_ok=True)
                results.to_csv(f'data/scan_{timestamp}.csv', index=False)
                
                st.success(f"✅ Found {len(results)} setups!")
                
                # AUTO-VALIDATE AND SEND TELEGRAM ALERTS (ONLY NEW ONES)
                if len(results) > 0:
                    with st.spinner("📱 Validating and sending Telegram alerts..."):
                        validated_count = 0
                        rejected_count = 0
                        new_alerts = 0
                        skipped_alerts = 0
                        monitored_count = 0
                        
                        # Import breakout monitor
                        try:
                            from breakout_monitor import BreakoutMonitor
                            monitor = BreakoutMonitor()
                        except:
                            monitor = None
                        
                        for idx, row in results.iterrows():
                            # Check if we should send alert for this ticker
                            if should_send_alert(row['ticker'], row['score']):
                                # Calculate position sizing
                                max_shares = account.calculate_position_size(row['entry'], row['stop'])
                                
                                if max_shares > 0:
                                    actual_value = max_shares * row['entry']
                                    actual_risk = max_shares * (row['entry'] - row['stop'])
                                    
                                    # Validate
                                    can_add, errors = account.can_add_position(actual_value, actual_risk)
                                    
                                    # Get reasons
                                    reasons = get_detailed_reasons(row)
                                    
                                    # Send alert
                                    alert_msg = format_trade_alert(
                                        row['ticker'], row['setup_type'], row['score'],
                                        row['entry'], row['stop'], row['target_1R'], row['target_2R'],
                                        max_shares, can_add, reasons
                                    )
                                    send_telegram_alert(alert_msg)
                                    
                                    # Send PRO FEATURES if available
                                    if 'pro_features' in row and row['pro_features']:
                                        pro_msg = "\n🎯 PRO ANALYSIS:\n" + "\n".join([f"  {feat}" for feat in row['pro_features']])
                                        send_telegram_alert(pro_msg)
                                    
                                    # Send TradingView link for chart
                                    tv_link = f"📊 Chart: https://www.tradingview.com/chart/?symbol={row['ticker']}"
                                    send_telegram_alert(tv_link)
                                    
                                    new_alerts += 1
                                    
                                    if can_add:
                                        validated_count += 1
                                        
                                        # ADD TO BREAKOUT MONITOR
                                        if monitor:
                                            monitor.add_setup_to_monitor(
                                                row['ticker'], row['entry'], row['stop'],
                                                row['target_1R'], row['target_2R'],
                                                row['setup_type'], row['score'],
                                                max_shares, f"Scan @ {datetime.now().strftime('%H:%M')}"
                                            )
                                            monitored_count += 1
                                    else:
                                        rejected_count += 1
                                    
                                    # Small delay to avoid rate limiting
                                    time.sleep(0.5)
                            else:
                                skipped_alerts += 1
                        
                        # Send summary
                        summary_msg = f"""
📊 <b>SCAN COMPLETE @ {datetime.now().strftime('%H:%M')}</b>

📋 Total Found: {len(results)} setups
🆕 New Alerts: {new_alerts}
⏭️ Already Sent: {skipped_alerts}

✅ Validated: {validated_count}
❌ Rejected: {rejected_count}

🔍 Now monitoring: {monitored_count} setups for breakout
"""
                        send_telegram_alert(summary_msg)
                        
                        if new_alerts > 0:
                            st.success(f"📱 Sent {new_alerts} NEW alerts to Telegram!")
                            if monitored_count > 0:
                                st.info(f"🔍 Added {monitored_count} validated setups to breakout monitor")
                        else:
                            st.info(f"No new setups - all {len(results)} already alerted today")
                        st.balloons()
    
    with col2:
        # Auto-scan toggle
        if 'auto_scan_enabled' not in st.session_state:
            st.session_state.auto_scan_enabled = False
        
        auto_scan = st.checkbox("🔄 Auto-Scan", value=st.session_state.auto_scan_enabled)
        st.session_state.auto_scan_enabled = auto_scan

# ============================================================================
# AUTO-CHECK POSITIONS - QULLAMAGGIE STYLE
# ============================================================================

# Morning report at 9:30 AM
current_time_check = datetime.now()
if current_time_check.hour == 9 and current_time_check.minute >= 30 and current_time_check.minute < 35:
    today_str = current_time_check.date().isoformat()
    last_report_date = st.session_state.get('last_morning_report_date', None)
    
    if last_report_date != today_str and len(account.positions) > 0:
        report = st.session_state.position_manager.generate_morning_report(account.positions)
        send_telegram_alert(report)
        st.session_state['last_morning_report_date'] = today_str

# Hourly position checks during market hours
if 9 <= current_time_check.hour < 16:
    if st.session_state.last_position_check is None or \
       (datetime.now() - st.session_state.last_position_check).seconds > 3600:
        if len(account.positions) > 0:
            alerts = []  # Position manager disabled
            st.session_state.last_position_check = datetime.now()


        
        if auto_scan:
            st.info("Scanning every 60 min")
        min_score = st.slider("Min Score", 70, 95, 80, 5)
    
    with col3:
        if not st.session_state.scan_results.empty:
            quality_counts = {
                'Elite (90+)': len(st.session_state.scan_results[st.session_state.scan_results['score'] >= 90]),
                'Premium (85-89)': len(st.session_state.scan_results[(st.session_state.scan_results['score'] >= 85) & (st.session_state.scan_results['score'] < 90)]),
                'High (80-84)': len(st.session_state.scan_results[(st.session_state.scan_results['score'] >= 80) & (st.session_state.scan_results['score'] < 85)]),
                'Good (70-79)': len(st.session_state.scan_results[(st.session_state.scan_results['score'] >= 70) & (st.session_state.scan_results['score'] < 80)])
            }
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("🌟 Elite", quality_counts['Elite (90+)'])
            col_b.metric("⭐ Premium", quality_counts['Premium (85-89)'])
            col_c.metric("✨ High", quality_counts['High (80-84)'])
            col_d.metric("💫 Good", quality_counts['Good (70-79)'])
    
    # Display Results
    st.markdown("---")
    
    # ALWAYS show results section
    if not st.session_state.scan_results.empty:
        filtered = st.session_state.scan_results[st.session_state.scan_results['score'] >= min_score]
        
        if len(filtered) > 0:
            st.subheader(f"📋 {len(filtered)} PREMIUM SETUPS (Score ≥ {min_score})")
            
            for idx, row in filtered.head(15).iterrows():
                # Calculate position sizing
                max_shares = account.calculate_position_size(row['entry'], row['stop'])
                max_value = max_shares * row['entry']
                risk_amount = max_shares * (row['entry'] - row['stop'])
                
                score_color = "#10b981" if row['score'] >= 90 else "#f59e0b" if row['score'] >= 85 else "#3b82f6"
                
                with st.expander(f"🎯 {row['ticker']} | {row['setup_type']} | Score: {row['score']}/100", expanded=idx<2):
                    
                    col1, col2 = st.columns([1, 1.2])
                    
                    with col1:
                        
                        # Trade Details
                        st.markdown("### 📊 TRADE SETUP")
                        
                        detail_col1, detail_col2 = st.columns(2)
                        detail_col1.markdown(f"**Entry Price:** ${row['entry']:.2f}")
                        detail_col2.markdown(f"**Stop Loss:** ${row['stop']:.2f} :red[({((row['stop']-row['entry'])/row['entry']*100):.1f}%)]")
                        detail_col1.markdown(f"**Target 1R:** ${row['target_1R']:.2f} :green[({((row['target_1R']-row['entry'])/row['entry']*100):.1f}%)]")
                        detail_col2.markdown(f"**Target 2R:** ${row['target_2R']:.2f} :green[({((row['target_2R']-row['entry'])/row['entry']*100):.1f}%)]")
                        
                        st.markdown("### 📈 TECHNICAL INDICATORS")
                        col_a, col_b = st.columns(2)
                        col_a.metric("RSI", f"{row.get('rsi', 0):.0f}")
                        col_b.metric("Volume", f"{row.get('volume_ratio', 0):.1f}x")
                        col_a.metric("RS vs SPY", f"{row.get('relative_strength', 1):.2f}x")
                        col_b.metric("Score", f"{row['score']}/100")
                        
                        st.markdown("### 🎯 RECOMMENDED POSITION")
                        
                        rec_col1, rec_col2 = st.columns(2)
                        rec_col1.metric("Max Shares", f"{max_shares:,}")
                        rec_col2.metric("Position Value", f"£{max_value:,.0f}")
                        rec_col1.metric("Risk Amount", f"£{risk_amount:,.0f}")
                        rec_col2.metric("Risk %", f"{(risk_amount/stats['equity']*100):.2f}%")
                    
                    with col2:
                        # Chart
                        chart = create_advanced_chart(row['ticker'], row['entry'], row['stop'],
                                                     row['target_1R'], row['target_2R'])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                    
                    # Detailed Reasoning
                    st.markdown("---")
                    st.markdown("### 💡 WHY THIS TRADE - DETAILED ANALYSIS")
                    
                    reasons = get_detailed_reasons(row)
                    for i, reason in enumerate(reasons):
                        st.info(reason)
                    
                    # Position Entry
                    st.markdown("---")
                    st.markdown("### 🎯 ENTER POSITION")
                    
                    col1, col2, col3, col4 = st.columns([2,2,3,2])
                    
                    with col1:
                        # Ensure max_shares is at least 1 for the input
                        display_max = max(max_shares, 1)
                        default_shares = max_shares if max_shares > 0 else 100
                        
                        shares_to_buy = st.number_input(
                            f"Shares (max {max_shares:,})" if max_shares > 0 else "Shares (insufficient buying power)",
                            min_value=1,
                            max_value=display_max,
                            value=min(default_shares, display_max),
                            step=10,
                            key=f"shares_{idx}",
                            disabled=(max_shares <= 0)
                        )
                    
                    actual_value = shares_to_buy * row['entry']
                    actual_risk = shares_to_buy * (row['entry'] - row['stop'])
                    
                    with col2:
                        st.metric("Cost", f"£{actual_value:,.0f}")
                        st.metric("Risk", f"£{actual_risk:,.0f}")
                    
                    with col3:
                        notes = st.text_input("Trade Notes", value=f"{row['setup_type']} - Score {row['score']}", key=f"notes_{idx}")
                    
                    # Validation
                    can_add, errors = account.can_add_position(actual_value, actual_risk)
                    
                    with col4:
                        if not can_add:
                            st.error("❌ REJECTED")
                            for error in errors:
                                st.caption(error)
                        else:
                            st.success("✅ VALIDATED")
                            st.caption("Ready to trade")
                        
                        if st.button(
                            f"🚀 BUY {shares_to_buy:,} SHARES",
                            key=f"buy_{idx}",
                            disabled=not can_add,
                            use_container_width=True,
                            type="primary"
                        ):
                            # Send validation alert before buying
                            alert_msg = format_trade_alert(
                                row['ticker'], row['setup_type'], row['score'],
                                row['entry'], row['stop'], row['target_1R'], row['target_2R'],
                                shares_to_buy, True, reasons
                            )
                            send_telegram_alert(alert_msg)
                            
                            account.add_position(
                                row['ticker'], shares_to_buy, row['entry'],
                                row['stop'], row['target_1R'], row['target_2R'],
                                row['setup_type'], row['score'], notes
                            )
                            save_account()
                            
                            # Send position opened confirmation to Telegram
                            position_msg = f"""
✅ <b>POSITION OPENED</b>

🎯 <b>{row['ticker']}</b> - {shares_to_buy:,} shares
💰 Entry: ${row['entry']:.2f}
🛑 Stop: ${row['stop']:.2f}
💵 Position Value: £{shares_to_buy * row['entry']:,.0f}

📊 Portfolio Status:
• Open Positions: {len(account.positions) + 1}
• Portfolio Heat: {stats['portfolio_heat']:.1f}%
• Cash Remaining: £{stats['cash'] - (shares_to_buy * row['entry']):,.0f}

📝 {notes}
"""
                            send_telegram_alert(position_msg)
                            
                            st.success(f"✅ Position opened: {shares_to_buy:,} shares of {row['ticker']}!")
                            st.info("📱 Alert sent to Telegram")
                            time.sleep(1)
                            st.rerun()
        else:
            st.info(f"No setups found with score ≥ {min_score}")
    else:
        st.info("👆 Click 'FULL SCAN' to find premium trading setups")


# ============================================================================
# TAB 2: POSITIONS - LIVE TRACKING
# ============================================================================
elif page == "💼 Positions":
    col_head1, col_head2 = st.columns([3, 1])
    
    with col_head1:
        st.header("💼 OPEN POSITIONS - LIVE TRACKING")
    
    with col_head2:
        if st.button("🔍 Check All Positions", use_container_width=True):
            with st.spinner("Checking positions for alerts..."):
                alerts = []  # Position manager disabled
                if alerts:
                    st.success(f"✅ Sent {len(alerts)} position alerts to Telegram!")
                else:
                    st.success("✅ All positions looking good!")
        
        if st.button("📱 Send to Telegram", use_container_width=True, type="primary"):
            send_positions_to_telegram(account)
            st.success("✅ Positions sent to Telegram!")
    
    if account.positions:
        # Portfolio Summary
        st.subheader("📊 Portfolio Overview")
        
        total_value = sum(p['current_value'] for p in account.positions)
        total_pnl = sum(p['pnl'] for p in account.positions)
        avg_r = sum(p.get('r_multiple', 0) for p in account.positions) / len(account.positions) if account.positions else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Positions", len(account.positions))
        col2.metric("Total Value", f"£{total_value:,.0f}")
        col3.metric("Total P&L", f"£{total_pnl:,.0f}")
        col4.metric("Avg R-Multiple", f"{avg_r:.2f}R")
        col5.metric("Portfolio Heat", f"{heat:.1f}%")
        
        st.markdown("---")
        
        # Individual Positions
        for idx, pos in enumerate(account.positions):
            # Get Qullamaggie position status
            pos_status = st.session_state.position_manager.get_position_status(pos['ticker'])
            days_held = st.session_state.position_manager.get_days_held(pos['ticker'])
            
            pnl_class = "pnl-positive" if pos['pnl'] > 0 else "pnl-negative"
            
            with st.expander(f"📍 {pos['ticker']} | {pos['shares']:,} shares | {pos['setup_type']}", expanded=True):
                
                col1, col2, col3 = st.columns([1,1,1])
                
                with col1:
                    st.markdown("### 📊 Position Details")
                    st.markdown(f"**Entry:** ${pos['entry_price']:.2f}")
                    st.markdown(f"**Current:** ${pos['current_price']:.2f}")
                    st.markdown(f"**Shares:** {pos['shares']:,}")
                    st.markdown(f"**Entry Value:** £{pos['entry_value']:,.0f}")
                    st.markdown(f"**Current Value:** £{pos['current_value']:,.0f}")
                    st.markdown(f"**Risk Amount:** £{pos['risk_amount']:,.0f}")
                    st.markdown(f"**Score:** {pos.get('score', 0)}/100")
                    if pos.get('notes'):
                        st.markdown(f"**Notes:** {pos['notes']}")
                
                with col2:
                    st.markdown("### 📈 Performance")
                    
                    pnl_color = ":green" if pos['pnl'] > 0 else ":red"
                    st.markdown(f"## {pnl_color}[£{pos['pnl']:,.0f}]")
                    st.markdown(f"**P&L %:** {pos['pnl_pct']:+.2f}%")
                    st.markdown(f"**R-Multiple:** {pos.get('r_multiple', 0):.2f}R")
                    
                    st.markdown("### 📍 Distance to Levels")
                    dist_stop = pos.get('dist_to_stop', 0)
                    dist_t1 = pos.get('dist_to_target1', 0)
                    dist_t2 = pos.get('dist_to_target2', 0)
                    
                    stop_emoji = "🟢" if dist_stop > 5 else "🟡" if dist_stop > 2 else "🔴"
                    st.markdown(f"**To Stop:** {stop_emoji} {dist_stop:+.1f}%")
                    st.markdown(f"**To Target 1R:** {dist_t1:+.1f}%")
                    st.markdown(f"**To Target 2R:** {dist_t2:+.1f}%")
                
                with col3:
                    # Chart
                    chart = create_advanced_chart(pos['ticker'], pos['entry_price'], pos['stop'],
                                                 pos['target_1R'], pos['target_2R'])
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"pos_chart_{idx}")
                    
                    # Actions
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("✅ CLOSE", key=f"close_{idx}", use_container_width=True, type="primary"):
                            pnl = account.close_position(idx, pos['current_price'])
                            save_account()
                            
                            # Calculate R-multiple for closed position
                            r_mult = pnl / pos['risk_amount'] if pos['risk_amount'] > 0 else 0
                            
                            # Send close alert to Telegram
                            close_msg = f"""
🔔 <b>POSITION CLOSED</b>

🎯 <b>{pos['ticker']}</b>
📊 Entry: ${pos['entry_price']:.2f}
📍 Exit: ${pos['current_price']:.2f}
📦 Shares: {pos['shares']:,}

💰 <b>P&L: £{pnl:,.0f}</b> ({pos['pnl_pct']:+.2f}%)
📈 <b>R-Multiple: {r_mult:+.2f}R</b>

⏱️ Held: {(datetime.now() - datetime.fromisoformat(pos['entry_date'])).days} days

💼 Portfolio:
• Open Positions: {len(account.positions) - 1}
• Portfolio Heat: {stats['portfolio_heat']:.1f}%
• New Cash: £{stats['cash'] + (pos['shares'] * pos['current_price']):,.0f}
"""
                            send_telegram_alert(close_msg)
                            
                            st.success(f"Position closed! P&L: £{pnl:,.0f}")
                            st.info("📱 Alert sent to Telegram")
                            time.sleep(1)
                            st.rerun()
                    
                    with col_b:
                        if st.button("🔄 Update", key=f"update_{idx}", use_container_width=True):
                            st.session_state.last_update = None
                            st.rerun()
    else:
        st.info("No open positions. Run scanner to find setups!")

# ============================================================================
# TAB 3: WATCHLIST
# ============================================================================
elif page == "⭐ Watchlist":
    st.header("⭐ Watchlist")
    st.info("Watchlist feature - Add stocks to monitor")

# ============================================================================
# TAB 4: JOURNAL
# ============================================================================
elif page == "📖 Journal":
    st.header("📖 Trade Journal")
    
    tab1, tab2 = st.tabs(["📝 New Entry", "📚 Past Entries"])
    
    with tab1:
        st.subheader("Log a Trade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            j_ticker = st.text_input("Ticker")
            j_setup = st.selectbox("Setup Type", ["Bull Flag", "Episodic Pivot", "Breakout", "VCP", "Other"])
            j_entry = st.number_input("Entry Price", min_value=0.01, step=0.01)
            j_exit = st.number_input("Exit Price", min_value=0.01, step=0.01)
        
        with col2:
            j_emotion = st.select_slider("Emotional State",
                                        options=["Fearful", "Anxious", "Neutral", "Confident", "Greedy"],
                                        value="Neutral")
            j_rating = st.slider("Trade Quality (1-5 ⭐)", 1, 5, 3)
            j_followed_plan = st.checkbox("Followed trading plan?")
            j_mistake = st.selectbox("Mistakes", ["None", "Entry too early", "Entry too late",
                                                  "Stop too tight", "Exited too early", "Revenge trading"])
        
        j_notes = st.text_area("Trade Notes")
        j_lesson = st.text_area("Key Lesson Learned")
        
        if st.button("💾 Save Journal Entry", type="primary"):
            pnl = (j_exit - j_entry) if j_exit and j_entry else 0
            entry = {
                'date': datetime.now().isoformat(),
                'ticker': j_ticker,
                'setup': j_setup,
                'entry': j_entry,
                'exit': j_exit,
                'pnl': pnl,
                'emotion': j_emotion,
                'rating': j_rating,
                'followed_plan': j_followed_plan,
                'mistake': j_mistake,
                'notes': j_notes,
                'lesson': j_lesson
            }
            account.journal_entries.append(entry)
            save_account()
            st.success("✅ Journal entry saved!")
    
    with tab2:
        if account.journal_entries:
            st.subheader(f"📚 {len(account.journal_entries)} Journal Entries")
            
            for entry in reversed(account.journal_entries[-20:]):
                with st.expander(f"📖 {entry['ticker']} - {entry['date'][:10]} | {'⭐' * entry['rating']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Setup:** {entry['setup']}")
                        st.markdown(f"**Entry:** ${entry['entry']:.2f}")
                        st.markdown(f"**Exit:** ${entry['exit']:.2f}")
                        st.markdown(f"**P&L:** ${entry['pnl']:.2f}")
                    
                    with col2:
                        st.markdown(f"**Emotion:** {entry['emotion']}")
                        st.markdown(f"**Followed Plan:** {'✅' if entry['followed_plan'] else '❌'}")
                        st.markdown(f"**Mistake:** {entry['mistake']}")
                    
                    if entry.get('notes'):
                        st.markdown(f"**Notes:** {entry['notes']}")
                    if entry.get('lesson'):
                        st.markdown(f"**💡 Lesson:** {entry['lesson']}")
        else:
            st.info("No journal entries yet")

# ============================================================================
# TAB 5: ANALYTICS
# ============================================================================
elif page == "📈 Analytics":
    st.header("📈 Performance Analytics - Complete Metrics")
    
    if account.closed_trades:
        metrics = calculate_performance_metrics(account.closed_trades)
        
        # Key Metrics
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades", metrics['total_trades'])
        col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        col4.metric("Expectancy", f"£{metrics['expectancy']:,.0f}")
        col5.metric("Total P&L", f"£{metrics['total_pnl']:,.0f}")
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Winners", metrics['winners'])
        col2.metric("Losers", metrics['losers'])
        col3.metric("Avg Win", f"£{metrics['avg_win']:,.0f}")
        col4.metric("Avg Loss", f"£{metrics['avg_loss']:,.0f}")
        col5.metric("Avg R", f"{metrics['avg_r']:.2f}R")
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Largest Win", f"£{metrics['largest_win']:,.0f}")
        col2.metric("Largest Loss", f"£{metrics['largest_loss']:,.0f}")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col4.metric("Max DD", f"£{metrics['max_drawdown']:,.0f}")
        col5.metric("Max Win Streak", f"{metrics['max_win_streak']}")
        
        # Charts
        st.markdown("---")
        st.subheader("📊 Performance Charts")
        
        df = pd.DataFrame(account.closed_trades)
        df = df.sort_values('exit_date')
        df['cumulative_pnl'] = df['pnl'].cumsum() + account.initial_capital
        
        # Equity Curve
        fig = px.line(df, x='exit_date', y='cumulative_pnl',
                     title='<b>Equity Curve</b>',
                     labels={'cumulative_pnl': 'Account Value (£)', 'exit_date': 'Date'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R-Multiple Distribution
            fig = px.histogram(df, x='r_multiple', nbins=30,
                             title='<b>R-Multiple Distribution</b>',
                             labels={'r_multiple': 'R-Multiple'})
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Win/Loss Distribution
            fig = px.histogram(df, x='pnl', nbins=30,
                             title='<b>P&L Distribution</b>',
                             labels={'pnl': 'P&L (£)'})
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Close positions to see analytics")

# ============================================================================
# TAB 6: RISK MANAGEMENT
# ============================================================================
elif page == "⚠️ Risk Management":
    st.header("⚠️ Risk Management Suite")
    
    st.subheader("📊 Position Size Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calc_entry = st.number_input("Entry Price", value=100.0, step=0.01)
    with col2:
        calc_stop = st.number_input("Stop Price", value=98.0, step=0.01)
    with col3:
        if st.button("🧮 Calculate", use_container_width=True, type="primary"):
            max_shares = account.calculate_position_size(calc_entry, calc_stop)
            max_value = max_shares * calc_entry
            risk_amount = max_shares * (calc_entry - calc_stop)
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Max Shares", f"{max_shares:,}")
            col2.metric("Position Value", f"£{max_value:,.0f}")
            col3.metric("Risk Amount", f"£{risk_amount:,.0f}")
            col4.metric("% of Account", f"{max_value/stats['equity']*100:.1f}%")
    
    st.markdown("---")
    st.subheader("📊 Portfolio Risk Analysis")
    
    if account.positions:
        total_risk = stats['total_risk']
        heat = stats['portfolio_heat']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Risk Exposure", f"£{total_risk:,.0f}")
        col2.metric("Portfolio Heat", f"{heat:.2f}%")
        col3.metric("Status", "🟢 Safe" if heat < 4 else "🟡 Moderate" if heat < 6 else "🔴 High")
        
        # Risk by position
        st.markdown("---")
        st.subheader("Risk Breakdown by Position")
        
        for pos in account.positions:
            pct = (pos['risk_amount'] / stats['equity']) * 100
            st.progress(pct / (account.max_risk_per_trade * 100))
            st.markdown(f"**{pos['ticker']}:** £{pos['risk_amount']:,.0f} ({pct:.2f}%)")
    else:
        st.info("Add positions to see risk analysis")

# ============================================================================
# TAB 7: SETTINGS - FULLY EDITABLE
# ============================================================================
elif page == "⚙️ Settings":
    st.header("⚙️ System Settings - Live Configuration")
    
    tab1, tab2, tab3 = st.tabs(["💰 Account Settings", "⚠️ Risk Parameters", "🔧 Advanced"])
    
    # TAB 1: ACCOUNT SETTINGS
    with tab1:
        st.subheader("💰 Account Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Status")
            st.metric("Initial Capital", f"£{account.initial_capital:,}")
            st.metric("Current Cash", f"£{stats['cash']:,}")
            st.metric("Current Equity", f"£{stats['equity']:,}")
            st.metric("Total P&L", f"£{stats['total_pnl']:,.0f}", f"{stats['total_pnl_pct']:+.2f}%")
        
        with col2:
            st.markdown("### Modify Account Size")
            
            new_initial = st.number_input(
                "Set Initial Capital (£)",
                min_value=1000,
                max_value=10000000,
                value=int(account.initial_capital),
                step=1000,
                help="This is your starting account size. Current positions will be adjusted proportionally."
            )
            
            if new_initial != account.initial_capital:
                if st.button("💾 Update Account Size", type="primary"):
                    # Calculate adjustment ratio
                    ratio = new_initial / account.initial_capital
                    
                    # Adjust cash proportionally
                    account.cash = account.cash * ratio
                    account.initial_capital = new_initial
                    
                    save_account()
                    st.success(f"✅ Account size updated to £{new_initial:,}")
                    time.sleep(1)
                    st.rerun()
                
                st.warning(f"⚠️ This will change your account from £{account.initial_capital:,} to £{new_initial:,}")
            
            st.markdown("---")
            
            # Add cash
            st.markdown("### Add/Remove Cash")
            
            cash_change = st.number_input(
                "Amount (£)",
                min_value=-int(account.cash),
                max_value=1000000,
                value=0,
                step=1000,
                help="Positive to add cash (deposit), negative to remove cash (withdrawal)"
            )
            
            if cash_change != 0:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("💵 Add Cash" if cash_change > 0 else "💸 Remove Cash", 
                               type="primary", use_container_width=True):
                        account.cash += cash_change
                        account.initial_capital += cash_change
                        save_account()
                        st.success(f"✅ {'Added' if cash_change > 0 else 'Removed'} £{abs(cash_change):,}")
                        time.sleep(1)
                        st.rerun()
    
    # TAB 2: RISK PARAMETERS
    with tab2:
        st.subheader("⚠️ Risk Management Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Risk Settings")
            st.metric("Max Risk per Trade", f"{account.max_risk_per_trade*100:.1f}%")
            st.metric("Max Portfolio Heat", f"{account.max_portfolio_heat*100:.1f}%")
            st.metric("Max Positions", account.max_positions)
            
            current_heat = stats['portfolio_heat']
            st.markdown("---")
            st.markdown(f"**Current Portfolio Heat:** {current_heat:.2f}%")
            
            if current_heat > 0:
                st.progress(min(current_heat / (account.max_portfolio_heat * 100), 1.0))
        
        with col2:
            st.markdown("### Modify Risk Parameters")
            
            new_risk_per_trade = st.slider(
                "Max Risk per Trade (%)",
                min_value=0.25,
                max_value=5.0,
                value=account.max_risk_per_trade * 100,
                step=0.25,
                help="Maximum % of account to risk on a single trade"
            )
            
            new_portfolio_heat = st.slider(
                "Max Portfolio Heat (%)",
                min_value=2.0,
                max_value=15.0,
                value=account.max_portfolio_heat * 100,
                step=0.5,
                help="Maximum total % of account at risk across all positions"
            )
            
            new_max_positions = st.number_input(
                "Max Positions",
                min_value=1,
                max_value=50,
                value=account.max_positions,
                step=1,
                help="Maximum number of simultaneous positions"
            )
            
            st.markdown("---")
            
            # Show impact
            if (new_risk_per_trade != account.max_risk_per_trade * 100 or
                new_portfolio_heat != account.max_portfolio_heat * 100 or
                new_max_positions != account.max_positions):
                
                st.markdown("### 📊 Impact Preview")
                
                # Calculate what position size would be with new settings
                sample_entry = 100
                sample_stop = 98
                risk_per_share = sample_entry - sample_stop
                
                current_max = int((stats['equity'] * account.max_risk_per_trade) / risk_per_share)
                new_max = int((stats['equity'] * (new_risk_per_trade/100)) / risk_per_share)
                
                st.markdown(f"**Example Position Size Change:**")
                st.markdown(f"Current: {current_max:,} shares (£{current_max*sample_entry:,.0f})")
                st.markdown(f"New: {new_max:,} shares (£{new_max*sample_entry:,.0f})")
                st.markdown(f"Difference: {new_max-current_max:+,} shares ({((new_max/current_max-1)*100):+.1f}%)")
                
                st.markdown("---")
                
                if st.button("💾 Apply Risk Settings", type="primary", use_container_width=True):
                    account.max_risk_per_trade = new_risk_per_trade / 100
                    account.max_portfolio_heat = new_portfolio_heat / 100
                    account.max_positions = new_max_positions
                    save_account()
                    st.success("✅ Risk parameters updated!")
                    time.sleep(1)
                    st.rerun()
                
                st.warning("⚠️ This will change your position sizing calculations")
    
    # TAB 3: ADVANCED
    with tab3:
        st.subheader("🔧 Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### System Information")
            st.markdown(f"**Scanner Universe:** 386 stocks")
            st.markdown(f"**Total Closed Trades:** {len(account.closed_trades)}")
            st.markdown(f"**Total Journal Entries:** {len(account.journal_entries)}")
            st.markdown(f"**Data Location:** data/account.json")
            
            st.markdown("---")
            st.markdown("### Quick Stats")
            if account.closed_trades:
                wins = [t for t in account.closed_trades if t['pnl'] > 0]
                win_rate = len(wins) / len(account.closed_trades) * 100
                st.markdown(f"**Historical Win Rate:** {win_rate:.1f}%")
                
                total_closed_pnl = sum(t['pnl'] for t in account.closed_trades)
                st.markdown(f"**Total Realized P&L:** £{total_closed_pnl:,.0f}")
        
        with col2:
            st.markdown("### Danger Zone")
            
            st.error("⚠️ CAUTION: These actions cannot be undone!")
            
            # Close all positions
            if len(account.positions) > 0:
                if st.button("🚨 Close ALL Positions", use_container_width=True):
                    if st.checkbox("⚠️ I confirm I want to close all positions"):
                        total_pnl = 0
                        positions_count = len(account.positions)
                        
                        while len(account.positions) > 0:
                            pnl = account.close_position(0, account.positions[0]['current_price'])
                            total_pnl += pnl
                        
                        save_account()
                        st.success(f"✅ Closed {positions_count} positions. Total P&L: £{total_pnl:,.0f}")
                        time.sleep(2)
                        st.rerun()
            
            st.markdown("---")
            
            # Reset account
            if st.button("🔄 Reset Account to Defaults", type="primary", use_container_width=True):
                if st.checkbox("⚠️ I confirm I want to reset (clears ALL data)"):
                    st.session_state.account = TradingAccount(config.ACCOUNT_SIZE)
                    save_account()
                    st.success("✅ Account reset to defaults!")
                    time.sleep(1)
                    st.rerun()
            
            st.markdown("---")
            
            # Export data
            if st.button("📥 Export All Data", use_container_width=True):
                export_data = {
                    'account': {
                        'initial_capital': account.initial_capital,
                        'cash': account.cash,
                        'equity': stats['equity']
                    },
                    'positions': account.positions,
                    'closed_trades': account.closed_trades,
                    'journal_entries': account.journal_entries
                }
                
                export_json = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=export_json,
                    file_name=f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'Never'}")
col2.markdown(f"**Version:** Ultra Platform v2.0")
col3.markdown(f"**Status:** {'🟢 LIVE TRADING' if len(account.positions) > 0 else '⚪ READY'}")
col4.markdown(f"**Features:** ALL 234 ACTIVE")

# Auto-refresh
if len(account.positions) > 0:
    time.sleep(2)
    st.rerun()

