"""
QULLAMAGGIE PRO SCANNER - Enhanced with TradingView Pro Features
Integrates all features from the Qullamaggie Trading System Pro indicator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import warnings
import logging
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QullamaggiePro:
    """
    Enhanced scanner with TradingView Pro features:
    1. 5-Star Quality Rating System
    2. VCP/Coiling Detection (Gold Dots)
    3. Relative Strength New Highs (Blue Dots)
    4. Pocket Pivots (Purple Dots)
    5. Professional Scoring Algorithm
    6. Advanced Pattern Recognition
    """
    
    def __init__(self, tickers: List[str], benchmark='SPY'):
        self.tickers = tickers
        self.benchmark = benchmark
        self.spy_data = None
        logger.info(f"🚀 Qullamaggie Pro Scanner initialized with {len(tickers)} tickers")
    
    def load_benchmark_data(self):
        """Load SPY/QQQ data for relative strength calculations"""
        try:
            self.spy_data = yf.download(self.benchmark, period='1y', interval='1d', progress=False)
            if len(self.spy_data) > 0:
                logger.info(f"✓ {self.benchmark} data loaded for RS calculations")
                return True
        except Exception as e:
            logger.error(f"Error loading {self.benchmark}: {e}")
        return False
    
    def calculate_5_star_rating(self, hist: pd.DataFrame, tech: Dict, rs_status: str) -> Tuple[int, Dict]:
        """
        5-STAR QUALITY RATING SYSTEM (TradingView Pro Feature)
        
        Scoring Algorithm:
        - Prior Move: 1.5 pts (50%+ in 60d = full points)
        - Above 50MA: 1.0 pt
        - RS New High: 1.0 pt (market leader)
        - Tightness: 1.0 pt (range < 3% = full points)
        - Volume Pattern: 1.0 pt (dry up → expansion)
        - MA Alignment: 0.5 pt (10>20>50>200)
        - Clean Setup: 0.5 pt (not choppy)
        
        Total: 6.5 points → 5 stars
        """
        score = 0.0
        details = {}
        
        current = hist.iloc[-1]
        current_price = current['Close']
        
        # 1. PRIOR MOVE (1.5 points max)
        if len(hist) >= 60:
            lookback_60d = hist.iloc[-60]['Close']
            prior_move_pct = ((current_price - lookback_60d) / lookback_60d) * 100
            details['prior_move'] = prior_move_pct
            
            if prior_move_pct >= 50:
                score += 1.5
            elif prior_move_pct >= 30:
                score += 1.0
            elif prior_move_pct >= 20:
                score += 0.5
        else:
            details['prior_move'] = 0
        
        # 2. ABOVE 50MA (1.0 point)
        ma50 = tech.get('ma_50', 0)
        above_50ma = current_price > ma50 if ma50 > 0 else False
        details['above_50ma'] = above_50ma
        if above_50ma:
            score += 1.0
        
        # 3. RS NEW HIGH - Market Leader (1.0 point)
        details['rs_status'] = rs_status
        if rs_status == "NEW HIGH":
            score += 1.0
        elif rs_status == "STRONG":
            score += 0.5
        
        # 4. TIGHTNESS (1.0 point)
        range_10d = hist['Close'].iloc[-10:] if len(hist) >= 10 else hist['Close']
        range_pct = ((range_10d.max() - range_10d.min()) / range_10d.min()) * 100
        details['range_10d'] = range_pct
        
        if range_pct < 3:
            score += 1.0
        elif range_pct < 5:
            score += 0.75
        elif range_pct < 10:
            score += 0.5
        elif range_pct < 15:
            score += 0.25
        
        # 5. VOLUME PATTERN (1.0 point)
        avg_vol_20d = hist['Volume'].iloc[-20:].mean()
        current_vol = current['Volume']
        recent_vol_trend = hist['Volume'].iloc[-5:].mean() / avg_vol_20d
        details['volume_pattern'] = recent_vol_trend
        
        # Dry up → expansion pattern
        if recent_vol_trend < 0.8 and current_vol > avg_vol_20d * 1.5:
            score += 1.0  # Perfect: dry up then explosion
        elif current_vol > avg_vol_20d * 1.5:
            score += 0.75  # Good: expansion
        elif recent_vol_trend < 0.8:
            score += 0.5  # Decent: drying up
        
        # 6. MA ALIGNMENT (0.5 points)
        ma10 = tech.get('ma_10', 0)
        ma20 = tech.get('ma_20', 0)
        ma200 = tech.get('ma_200', 0)
        
        perfect_alignment = (ma10 > ma20 > ma50 > ma200) if all([ma10, ma20, ma50, ma200]) else False
        details['ma_alignment'] = perfect_alignment
        if perfect_alignment:
            score += 0.5
        elif ma10 > ma20 > ma50:
            score += 0.25
        
        # 7. CLEAN SETUP (0.5 points) - Not choppy
        volatility = hist['Close'].iloc[-20:].std() / hist['Close'].iloc[-20:].mean()
        details['volatility'] = volatility
        if volatility < 0.02:  # Very smooth
            score += 0.5
        elif volatility < 0.035:  # Acceptable
            score += 0.25
        
        # Convert to stars (0-5)
        # 6.5 points max → 5 stars
        # Each star = 1.3 points
        stars = min(5, int((score / 1.3) + 0.5))
        
        details['total_score'] = score
        details['stars'] = stars
        
        return stars, details
    
    def detect_vcp_coiling(self, hist: pd.DataFrame) -> Optional[Dict]:
        """
        VCP/COILING DETECTION - Gold Dots (TradingView Pro Feature)
        
        Volatility Contraction Pattern:
        - Price range tightening (<10%)
        - Volume drying up
        - Higher lows pattern
        - Surfing 10MA or 20MA
        """
        if len(hist) < 20:
            return None
        
        current = hist.iloc[-1]
        last_10 = hist.iloc[-10:]
        
        # Calculate range
        range_high = last_10['High'].max()
        range_low = last_10['Low'].min()
        range_pct = ((range_high - range_low) / range_low) * 100
        
        # Volume drying up?
        avg_vol_20 = hist['Volume'].iloc[-20:].mean()
        recent_vol = last_10['Volume'].mean()
        vol_ratio = recent_vol / avg_vol_20
        
        # Surfing which MA?
        ma10 = hist['Close'].rolling(10).mean().iloc[-1]
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        current_price = current['Close']
        
        surfing_ma = None
        if abs(current_price - ma10) / current_price < 0.02:  # Within 2%
            surfing_ma = "10MA"
        elif abs(current_price - ma20) / current_price < 0.03:  # Within 3%
            surfing_ma = "20MA"
        
        # Check for higher lows
        lows = last_10['Low'].values
        higher_lows = all(lows[i] >= lows[i-1] * 0.98 for i in range(1, len(lows)))
        
        # VCP criteria
        is_coiling = (
            range_pct < 10 and
            vol_ratio < 0.9 and
            surfing_ma is not None and
            higher_lows
        )
        
        if is_coiling:
            return {
                'type': 'COILING',
                'range_pct': range_pct,
                'vol_ratio': vol_ratio,
                'surfing': surfing_ma,
                'message': f"VCP forming | Range: {range_pct:.1f}% | Vol: {vol_ratio:.2f}x | Surfing {surfing_ma}"
            }
        
        return None
    
    def detect_rs_new_high(self, hist: pd.DataFrame) -> str:
        """
        RELATIVE STRENGTH NEW HIGH - Blue Dots (TradingView Pro Feature)
        
        Detects when RS line makes new 50-day high
        = Market leader BEFORE breakout
        """
        if self.spy_data is None or len(hist) < 50:
            return "UNKNOWN"
        
        try:
            # Calculate RS line
            stock_returns = hist['Close'].pct_change()
            spy_returns = self.spy_data['Close'].pct_change()
            
            # Align dates
            common_dates = stock_returns.index.intersection(spy_returns.index)
            if len(common_dates) < 50:
                return "UNKNOWN"
            
            stock_returns = stock_returns[common_dates]
            spy_returns = spy_returns[common_dates]
            
            # RS = Stock performance / SPY performance
            rs_line = (1 + stock_returns).cumprod() / (1 + spy_returns).cumprod()
            
            if len(rs_line) < 50:
                return "UNKNOWN"
            
            current_rs = rs_line.iloc[-1]
            max_rs_50d = rs_line.iloc[-50:].max()
            
            # Check if at new high
            if current_rs >= max_rs_50d * 0.99:  # Within 1% of high
                return "NEW HIGH"
            elif current_rs >= max_rs_50d * 0.95:  # Within 5%
                return "STRONG"
            else:
                return "WEAK"
        
        except Exception as e:
            return "UNKNOWN"
    
    def detect_pocket_pivot(self, hist: pd.DataFrame) -> Optional[Dict]:
        """
        POCKET PIVOT - Purple Dots (TradingView Pro Feature)
        
        Institutional accumulation inside the base
        Volume > Highest down-volume of last 10 days
        """
        if len(hist) < 10:
            return None
        
        current = hist.iloc[-1]
        last_10 = hist.iloc[-10:-1]  # Exclude today
        
        # Find highest down-volume day
        down_days = last_10[last_10['Close'] < last_10['Open']]
        if len(down_days) == 0:
            return None
        
        max_down_vol = down_days['Volume'].max()
        current_vol = current['Volume']
        
        # Is today an up day?
        is_up_day = current['Close'] > current['Open']
        
        # Pocket Pivot criteria
        if is_up_day and current_vol > max_down_vol:
            vol_comparison = current_vol / max_down_vol
            return {
                'type': 'POCKET_PIVOT',
                'current_vol': current_vol,
                'max_down_vol': max_down_vol,
                'ratio': vol_comparison,
                'message': f"Pocket Pivot! Vol: {vol_comparison:.1f}x highest down-day"
            }
        
        return None
    
    def calculate_adr(self, hist: pd.DataFrame, period: int = 20) -> float:
        """
        Average Daily Range - Volatility assessment
        """
        if len(hist) < period:
            return 0
        
        last_n = hist.iloc[-period:]
        daily_ranges = ((last_n['High'] - last_n['Low']) / last_n['Low']) * 100
        return daily_ranges.mean()
    
    def scan_ticker_pro(self, ticker: str) -> List[Dict]:
        """
        PRO SCAN with all TradingView Pro features
        """
        setups = []
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo', interval='1d')
            
            if len(hist) < 60:
                return []
            
            # Calculate basic technicals
            current = hist.iloc[-1]
            current_price = current['Close']
            
            tech = {
                'ma_10': hist['Close'].rolling(10).mean().iloc[-1],
                'ma_20': hist['Close'].rolling(20).mean().iloc[-1],
                'ma_50': hist['Close'].rolling(50).mean().iloc[-1],
                'ma_200': hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else 0,
                'rsi': self._calculate_rsi(hist['Close']),
                'atr': hist['High'].rolling(14).max() - hist['Low'].rolling(14).min(),
            }
            
            # PRO FEATURES
            
            # 1. Relative Strength Status
            rs_status = self.detect_rs_new_high(hist)
            
            # 2. VCP/Coiling Detection
            coiling = self.detect_vcp_coiling(hist)
            
            # 3. Pocket Pivot Detection
            pocket_pivot = self.detect_pocket_pivot(hist)
            
            # 4. 5-Star Quality Rating
            stars, rating_details = self.calculate_5_star_rating(hist, tech, rs_status)
            
            # 5. ADR Calculation
            adr = self.calculate_adr(hist)
            
            # Detect main setups (Breakout, EP, Parabolic)
            setup = self._detect_main_setup(hist, tech)
            
            if setup:
                # Enhance with Pro features
                setup['ticker'] = ticker
                setup['stars'] = stars
                setup['rating_details'] = rating_details
                setup['rs_status'] = rs_status
                setup['coiling'] = coiling
                setup['pocket_pivot'] = pocket_pivot
                setup['adr'] = adr
                setup['current_price'] = current_price
                
                # Pro Dashboard Data
                setup['dashboard'] = {
                    'prior_move_60d': rating_details.get('prior_move', 0),
                    'above_50ma': rating_details.get('above_50ma', False),
                    'setup_quality': '⭐' * stars,
                    'adr_20': adr,
                    'trend': 'BULLISH 🟢' if rating_details.get('ma_alignment', False) else 'MIXED 🟡',
                    'rs_vs_index': self._format_rs_status(rs_status),
                    'status': self._get_status_emoji(setup, coiling),
                    'volume': self._get_volume_status(rating_details.get('volume_pattern', 1)),
                    'range_10d': rating_details.get('range_10d', 0)
                }
                
                setups.append(setup)
        
        except Exception as e:
            pass
        
        return setups
    
    def _detect_main_setup(self, hist: pd.DataFrame, tech: Dict) -> Optional[Dict]:
        """Detect Breakout, EP, or Parabolic setup"""
        current = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        # EPISODIC PIVOT
        gap_pct = ((current['Open'] - prev['Close']) / prev['Close']) * 100
        vol_ratio = current['Volume'] / hist['Volume'].iloc[-20:].mean()
        
        if gap_pct >= 10 and vol_ratio >= 2.0:
            return {
                'setup_type': 'EP',
                'entry': current['High'],
                'stop': current['Low'],
                'target_1R': current['High'] + (current['High'] - current['Low']),
                'target_2R': current['High'] + 2 * (current['High'] - current['Low']),
                'gap_pct': gap_pct,
                'volume_ratio': vol_ratio
            }
        
        # BREAKOUT
        consol_high = hist['High'].iloc[-20:].max()
        if current['Close'] > consol_high and vol_ratio >= 1.5:
            return {
                'setup_type': 'BREAKOUT',
                'entry': consol_high,
                'stop': hist['Low'].iloc[-10:].min(),
                'target_1R': consol_high + (consol_high - hist['Low'].iloc[-10:].min()),
                'target_2R': consol_high + 2 * (consol_high - hist['Low'].iloc[-10:].min()),
                'volume_ratio': vol_ratio
            }
        
        # PARABOLIC SHORT
        if tech['rsi'] > 75 and len(hist) >= 3:
            up_days = (hist['Close'].iloc[-3:] > hist['Close'].shift(1).iloc[-3:]).sum()
            if up_days >= 3:
                return {
                    'setup_type': 'PARABOLIC_SHORT',
                    'entry': current['Close'],
                    'stop': current['High'] * 1.03,
                    'target_1R': current['Close'] * 0.95,
                    'target_2R': current['Close'] * 0.90,
                    'rsi': tech['rsi']
                }
        
        return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def _format_rs_status(self, rs_status: str) -> str:
        """Format RS status with emojis"""
        if rs_status == "NEW HIGH":
            return "NEW HIGH 💎"
        elif rs_status == "STRONG":
            return "STRONG 💪"
        elif rs_status == "WEAK":
            return "WEAK 📉"
        return "UNKNOWN"
    
    def _get_status_emoji(self, setup: Dict, coiling: Optional[Dict]) -> str:
        """Get status with emoji"""
        if setup['setup_type'] == 'BREAKOUT':
            return "BREAKOUT 🚀"
        elif setup['setup_type'] == 'EP':
            return "EPISODIC ⚡"
        elif setup['setup_type'] == 'PARABOLIC_SHORT':
            return "PARABOLIC 🔻"
        elif coiling:
            return "COILING 🕸️"
        return "WAITING ⏳"
    
    def _get_volume_status(self, vol_pattern: float) -> str:
        """Get volume status"""
        if vol_pattern > 1.3:
            return "EXPANSION 🔊"
        elif vol_pattern < 0.8:
            return "DRY UP 🔇"
        return "NORMAL"
    
    def run_pro_scan(self, max_workers: int = 10) -> pd.DataFrame:
        """
        Run full scan with all Pro features
        """
        logger.info("=" * 80)
        logger.info("🚀 QULLAMAGGIE PRO SCANNER - TradingView Edition")
        logger.info("=" * 80)
        logger.info(f"Scanning {len(self.tickers)} tickers...")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load benchmark data for RS calculations
        self.load_benchmark_data()
        
        all_setups = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.scan_ticker_pro, ticker): ticker 
                      for ticker in self.tickers}
            
            completed = 0
            for future in futures:
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Progress: {completed}/{len(self.tickers)} tickers scanned...")
                
                try:
                    setups = future.result(timeout=30)
                    all_setups.extend(setups)
                except Exception:
                    pass
        
        if not all_setups:
            logger.info("No setups found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_setups)
        
        # Extract sortable score from rating_details
        df['total_score'] = df['rating_details'].apply(lambda x: x.get('total_score', 0) if isinstance(x, dict) else 0)
        
        # Sort by stars (highest first), then by total score
        df = df.sort_values(['stars', 'total_score'], ascending=[False, False])
        
        logger.info("=" * 80)
        logger.info(f"📊 SCAN COMPLETE: {len(df)} SETUPS FOUND")
        logger.info("=" * 80)
        logger.info(f"\n5-STAR BREAKDOWN:")
        for stars in range(5, 0, -1):
            count = len(df[df['stars'] == stars])
            if count > 0:
                logger.info(f"  {'⭐' * stars} ({stars}-Star): {count} setups")
        
        return df


# Compatibility with existing scanner
class UltraScannerEngine(QullamaggiePro):
    """Wrapper for compatibility"""
    def __init__(self, tickers: List[str]):
        super().__init__(tickers)
    
    def run_full_scan(self, max_workers: int = 10) -> pd.DataFrame:
        return self.run_pro_scan(max_workers)
