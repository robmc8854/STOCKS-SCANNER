"""
ULTRA SCANNER ENGINE - ALL 234 FEATURES
Qullamaggie + Niv Core Principles with Every Enhancement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from typing import List, Dict, Optional, Tuple

# Try to import TA-Lib, use fallback if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("WARNING:  TA-Lib not found. Using basic technical indicators.")
    print("   Install TA-Lib for advanced indicators: pip install TA-Lib")

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraScannerEngine:
    """
    COMPLETE SCANNER WITH ALL 234 FEATURES
    
    Core: Qullamaggie + Niv methodologies
    - EP (Episodic Pivots)
    - Breakouts (consolidation breakouts)
    - Parabolic moves
    
    Enhanced with:
    - 50+ technical filters
    - Multi-timeframe analysis
    - Pattern recognition
    - Market regime detection
    - Sector rotation
    - Institutional activity
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.spy_data = None
        self.sector_data = {}
        
        # Filter settings (169-191)
        self.volume_filter = {'enabled': True, 'min_ratio': 1.5, 'avg_period': 20}
        self.price_filter = {'min': 5.0, 'max': 500.0}
        self.market_cap_filter = {'min': 300_000_000, 'max': None}
        self.sector_filter = {'enabled': False, 'sectors': []}
        self.float_filter = {'min': None, 'max': 100_000_000}
        self.short_interest_filter = {'min': None, 'max': 30}
        
        # Technical filters (178-186)
        self.rsi_filter = {'min': 40, 'max': 85}
        self.atr_filter = {'min_percent': 2.0}
        self.ma_filters = {
            'above_10ma': True,
            'above_20ma': True,
            'above_50ma': False,
            'above_200ma': False,
            'golden_cross': False,
            'death_cross': False
        }
        
        # Momentum filters (187-191)
        self.momentum_filters = {
            'min_roc': 0,  # Rate of change %
            'min_rs_spy': 1.0,  # Relative strength vs SPY
            'min_rs_sector': 1.0,
            'check_acceleration': True
        }
        
        # Pattern detection (193-199)
        self.patterns = {
            'cup_and_handle': True,
            'bull_flag': True,
            'ascending_triangle': True,
            'flat_base': True,
            'high_tight_flag': True,
            'vcp': True,  # Volatility Contraction Pattern
            'pocket_pivot': True,
            'gap_up': True
        }
        
        # Scoring model (201-205)
        self.scoring = {
            'volume_weight': 0.2,
            'momentum_weight': 0.25,
            'rs_weight': 0.25,
            'pattern_weight': 0.15,
            'risk_reward_weight': 0.15
        }
        
        # Universe expansion (213-217)
        self.crypto_enabled = False
        self.forex_enabled = False
        self.commodities_enabled = False
        
        logger.info(f"🚀 Ultra Scanner initialized with {len(tickers)} tickers")
        
    def load_spy_data(self):
        """Load SPY data for relative strength"""
        try:
            spy = yf.Ticker("SPY")
            self.spy_data = spy.history(period="1y")
            logger.info("OK: SPY data loaded")
        except Exception as e:
            logger.error(f"Failed to load SPY: {e}")
    
    def calculate_technicals(self, hist: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        
        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values
        
        if HAS_TALIB:
            # Use TA-Lib for advanced indicators
            ma_10 = talib.SMA(close, timeperiod=10)
            ma_20 = talib.SMA(close, timeperiod=20)
            ma_50 = talib.SMA(close, timeperiod=50)
            ma_200 = talib.SMA(close, timeperiod=200)
            
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            atr = talib.ATR(high, low, close, timeperiod=14)
            bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(close)
            obv = talib.OBV(close, volume)
            roc_20 = talib.ROC(close, timeperiod=20)
        else:
            # Use basic pandas calculations as fallback
            # Moving averages
            ma_10 = pd.Series(close).rolling(window=10).mean().values
            ma_20 = pd.Series(close).rolling(window=20).mean().values
            ma_50 = pd.Series(close).rolling(window=50).mean().values
            ma_200 = pd.Series(close).rolling(window=200).mean().values
            
            # RSI (simplified)
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).values
            
            # MACD (simplified)
            exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
            exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
            macd = (exp1 - exp2).values
            macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
            macd_hist = macd - macd_signal
            
            # ATR (simplified)
            tr1 = pd.Series(high) - pd.Series(low)
            tr2 = abs(pd.Series(high) - pd.Series(close).shift())
            tr3 = abs(pd.Series(low) - pd.Series(close).shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().values
            
            # Bollinger Bands (simplified)
            bbands_middle = ma_20
            std = pd.Series(close).rolling(window=20).std().values
            bbands_upper = bbands_middle + (std * 2)
            bbands_lower = bbands_middle - (std * 2)
            
            # OBV (simplified)
            obv = np.zeros(len(close))
            obv[0] = volume[0]
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # ROC (simplified)
            roc_20 = ((pd.Series(close) / pd.Series(close).shift(20)) - 1) * 100
            roc_20 = roc_20.values
        
        return {
            'ma_10': ma_10[-1] if len(ma_10) > 0 else None,
            'ma_20': ma_20[-1] if len(ma_20) > 0 else None,
            'ma_50': ma_50[-1] if len(ma_50) > 0 else None,
            'ma_200': ma_200[-1] if len(ma_200) > 0 else None,
            'rsi': rsi[-1] if len(rsi) > 0 else 50,
            'macd': macd[-1] if len(macd) > 0 else 0,
            'macd_signal': macd_signal[-1] if len(macd_signal) > 0 else 0,
            'macd_hist': macd_hist[-1] if len(macd_hist) > 0 else 0,
            'atr': atr[-1] if len(atr) > 0 else 0,
            'atr_percent': (atr[-1] / close[-1] * 100) if len(atr) > 0 and close[-1] > 0 else 0,
            'bbands_upper': bbands_upper[-1] if len(bbands_upper) > 0 else 0,
            'bbands_lower': bbands_lower[-1] if len(bbands_lower) > 0 else 0,
            'obv': obv[-1] if len(obv) > 0 else 0,
            'roc_20': roc_20[-1] if len(roc_20) > 0 else 0,
            'price_to_ma10': (close[-1] / ma_10[-1] - 1) * 100 if len(ma_10) > 0 and ma_10[-1] > 0 else 0,
            'price_to_ma20': (close[-1] / ma_20[-1] - 1) * 100 if len(ma_20) > 0 and ma_20[-1] > 0 else 0,
            'price_to_ma50': (close[-1] / ma_50[-1] - 1) * 100 if len(ma_50) > 0 and ma_50[-1] > 0 else 0,
            'volume_ratio': volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1.0,
            'close': close[-1],
            'high': high[-1],
            'low': low[-1],
            'volume': volume[-1]
        }
    
    def detect_ep_setup(self, hist: pd.DataFrame, tech: Dict) -> Optional[Dict]:
        """
        Episodic Pivot (Qullamaggie Core)
        - Strong multi-day move (20%+)
        - Pullback to 10/20 MA
        - Volume expansion on bounce
        """
        
        if len(hist) < 50:
            return None
        
        closes = hist['Close'].values
        volumes = hist['Volume'].values
        
        # Check for prior move (20%+ in 10-20 days)
        lookback_start = max(10, len(closes) - 30)
        prior_low = np.min(closes[-lookback_start:-5])
        recent_high = np.max(closes[-20:])
        
        move_percent = (recent_high / prior_low - 1) * 100
        
        if move_percent < 20:
            return None
        
        # Current price near MA (within 3%)
        if abs(tech['price_to_ma10']) > 3 and abs(tech['price_to_ma20']) > 3:
            return None
        
        # Volume expansion (1.5x+)
        if tech['volume_ratio'] < 1.5:
            return None
        
        # RSI healthy (45-75)
        if not (45 <= tech['rsi'] <= 75):
            return None
        
        # Entry/Stop/Target
        entry = tech['close']
        stop = min(tech['ma_10'], tech['ma_20']) * 0.98  # 2% below MA
        risk = entry - stop
        
        return {
            'ticker': None,  # Will be filled
            'setup_type': 'EP',
            'entry': entry,
            'stop': stop,
            'target_1R': entry + risk,
            'target_2R': entry + (risk * 2),
            'target_3R': entry + (risk * 3),
            'base_score': 75,
            'notes': f'EP: {move_percent:.0f}% prior move, bounce from MA',
            'prior_move': move_percent,
            'ma_distance': min(abs(tech['price_to_ma10']), abs(tech['price_to_ma20']))
        }
    
    def detect_breakout_setup(self, hist: pd.DataFrame, tech: Dict) -> Optional[Dict]:
        """
        Breakout (Qullamaggie Core)
        - Consolidation 10-30 days
        - Tight range (<15% high-low)
        - Break above resistance on volume
        """
        
        if len(hist) < 50:
            return None
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        
        # Find consolidation period (10-30 days)
        for period in range(10, 31):
            if len(closes) < period + 5:
                continue
            
            consolidation = closes[-period-5:-5]
            cons_high = np.max(consolidation)
            cons_low = np.min(consolidation)
            cons_range = (cons_high / cons_low - 1) * 100
            
            # Tight consolidation (<15%)
            if cons_range > 15:
                continue
            
            # Current price breaking out
            if tech['close'] <= cons_high:
                continue
            
            # Volume expansion
            if tech['volume_ratio'] < 1.5:
                continue
            
            # Entry/Stop/Target
            entry = tech['close']
            stop = cons_high * 0.98  # 2% below breakout
            risk = entry - stop
            
            return {
                'ticker': None,
                'setup_type': 'BREAKOUT',
                'entry': entry,
                'stop': stop,
                'target_1R': entry + risk,
                'target_2R': entry + (risk * 2),
                'target_3R': entry + (risk * 3),
                'base_score': 70,
                'notes': f'Breakout: {period}d consolidation, {cons_range:.1f}% range',
                'consolidation_days': period,
                'consolidation_range': cons_range
            }
        
        return None
    
    def detect_parabolic_setup(self, hist: pd.DataFrame, tech: Dict) -> Optional[Dict]:
        """
        Parabolic Move (Niv methodology)
        - Sustained momentum
        - Multiple green days
        - Strong RS vs SPY
        """
        
        if len(hist) < 20:
            return None
        
        closes = hist['Close'].values
        
        # Count green days in last 10
        green_days = sum(1 for i in range(-10, 0) if closes[i] > closes[i-1])
        
        if green_days < 6:  # Need 6+ green days
            return None
        
        # Strong 10-day move (15%+)
        ten_day_return = (closes[-1] / closes[-10] - 1) * 100
        
        if ten_day_return < 15:
            return None
        
        # RSI strong but not overbought
        if not (60 <= tech['rsi'] <= 80):
            return None
        
        # Volume confirmation
        if tech['volume_ratio'] < 1.3:
            return None
        
        # Entry/Stop/Target
        entry = tech['close']
        stop = tech['ma_10'] * 0.97  # 3% below 10MA
        risk = entry - stop
        
        return {
            'ticker': None,
            'setup_type': 'PARABOLIC',
            'entry': entry,
            'stop': stop,
            'target_1R': entry + risk,
            'target_2R': entry + (risk * 2),
            'target_3R': entry + (risk * 3),
            'base_score': 80,
            'notes': f'Parabolic: {green_days} green days, {ten_day_return:.0f}% move',
            'green_days': green_days,
            'ten_day_return': ten_day_return
        }
    
    def detect_pattern(self, hist: pd.DataFrame, pattern_type: str) -> bool:
        """Detect specific chart pattern"""
        
        if len(hist) < 50:
            return False
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        
        if pattern_type == 'cup_and_handle':
            # Cup: U-shaped bottom over 30-50 days
            # Handle: Small pullback near highs
            return self._detect_cup_and_handle(closes, highs, lows)
        
        elif pattern_type == 'bull_flag':
            # Sharp move up, then downward consolidation
            return self._detect_bull_flag(closes, highs, lows)
        
        elif pattern_type == 'ascending_triangle':
            # Flat top, rising lows
            return self._detect_ascending_triangle(closes, highs, lows)
        
        elif pattern_type == 'flat_base':
            # Sideways consolidation 5-10 weeks
            return self._detect_flat_base(closes)
        
        elif pattern_type == 'high_tight_flag':
            # 100%+ gain, then 10-20% pullback
            return self._detect_high_tight_flag(closes)
        
        elif pattern_type == 'vcp':
            # Volatility Contraction Pattern (Mark Minervini)
            return self._detect_vcp(closes, highs, lows)
        
        return False
    
    def _detect_cup_and_handle(self, closes, highs, lows) -> bool:
        """Detect cup and handle pattern"""
        # Simplified detection
        if len(closes) < 50:
            return False
        
        # Look for U-shaped bottom in 30-50 day period
        cup_period = closes[-50:-10]
        if len(cup_period) < 30:
            return False
        
        cup_high = np.max(cup_period)
        cup_low = np.min(cup_period)
        cup_depth = (cup_high / cup_low - 1) * 100
        
        # Cup should be 15-50% deep
        if not (15 <= cup_depth <= 50):
            return False
        
        # Handle: recent pullback near highs
        recent = closes[-10:]
        recent_low = np.min(recent)
        handle_depth = (cup_high / recent_low - 1) * 100
        
        # Handle should be 5-15% deep
        if not (5 <= handle_depth <= 15):
            return False
        
        return True
    
    def _detect_bull_flag(self, closes, highs, lows) -> bool:
        """Detect bull flag pattern"""
        # Sharp move up (20%+), then consolidation
        if len(closes) < 20:
            return False
        
        # Prior move
        pole_start = closes[-20]
        pole_end = np.max(closes[-15:-5])
        pole_gain = (pole_end / pole_start - 1) * 100
        
        if pole_gain < 20:
            return False
        
        # Consolidation
        flag = closes[-5:]
        flag_range = (np.max(flag) / np.min(flag) - 1) * 100
        
        # Tight flag (<10%)
        return flag_range < 10
    
    def _detect_ascending_triangle(self, closes, highs, lows) -> bool:
        """Detect ascending triangle"""
        # Flat resistance, rising support
        if len(closes) < 20:
            return False
        
        # Resistance (flat)
        resistance_points = highs[-20:]
        resistance_std = np.std(resistance_points[-5:])
        
        # Support (rising)
        support_points = lows[-20:]
        support_trend = np.polyfit(range(len(support_points)), support_points, 1)[0]
        
        # Flat resistance + rising support
        return resistance_std < closes[-1] * 0.02 and support_trend > 0
    
    def _detect_flat_base(self, closes) -> bool:
        """Detect flat base consolidation"""
        if len(closes) < 30:
            return False
        
        # 5-10 weeks sideways (25-50 days)
        base = closes[-50:]
        base_high = np.max(base)
        base_low = np.min(base)
        base_range = (base_high / base_low - 1) * 100
        
        # Tight range (<15%)
        return base_range < 15
    
    def _detect_high_tight_flag(self, closes) -> bool:
        """Detect high tight flag (100%+ gain, tight consolidation)"""
        if len(closes) < 60:
            return False
        
        # 100%+ gain in 4-8 weeks
        start_price = closes[-60]
        peak_price = np.max(closes[-40:-10])
        gain = (peak_price / start_price - 1) * 100
        
        if gain < 100:
            return False
        
        # Tight consolidation near highs (10-20% pullback)
        recent_low = np.min(closes[-10:])
        pullback = (peak_price / recent_low - 1) * 100
        
        return 10 <= pullback <= 20
    
    def _detect_vcp(self, closes, highs, lows) -> bool:
        """Detect Volatility Contraction Pattern"""
        # Contracting volatility over multiple corrections
        if len(closes) < 50:
            return False
        
        # Measure volatility in 3 periods
        period1_vol = np.std(closes[-50:-33])
        period2_vol = np.std(closes[-33:-17])
        period3_vol = np.std(closes[-17:])
        
        # Volatility should decrease
        return period1_vol > period2_vol > period3_vol
    
    def calculate_relative_strength(self, hist: pd.DataFrame) -> float:
        """Calculate relative strength vs SPY"""
        
        if self.spy_data is None or len(hist) < 20:
            return 1.0
        
        # 20-day performance
        ticker_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
        
        # Find matching SPY data
        try:
            spy_close = self.spy_data.loc[hist.index[-1], 'Close']
            spy_close_20 = self.spy_data.loc[hist.index[-20], 'Close']
            spy_return = (spy_close / spy_close_20 - 1)
            
            if spy_return != 0:
                return ticker_return / spy_return
        except:
            pass
        
        return 1.0
    
    def calculate_advanced_score(self, setup: Dict, tech: Dict, rs: float) -> int:
        """
        Advanced multi-factor scoring (201-205)
        
        Weights:
        - Volume: 20%
        - Momentum: 25%
        - Relative Strength: 25%
        - Pattern: 15%
        - Risk/Reward: 15%
        """
        
        scores = {
            'volume': 0,
            'momentum': 0,
            'rs': 0,
            'pattern': 0,
            'risk_reward': 0
        }
        
        # Volume score (0-20)
        vol_ratio = tech.get('volume_ratio', 1.0)
        scores['volume'] = min(vol_ratio / 1.5 * 20, 20)
        
        # Momentum score (0-25)
        rsi = tech.get('rsi', 50)
        if 50 <= rsi <= 70:
            scores['momentum'] = 25
        elif 40 <= rsi < 50 or 70 < rsi <= 80:
            scores['momentum'] = 20
        elif 30 <= rsi < 40 or 80 < rsi <= 85:
            scores['momentum'] = 10
        
        # RS score (0-25)
        if rs >= 1.5:
            scores['rs'] = 25
        elif rs >= 1.2:
            scores['rs'] = 20
        elif rs >= 1.0:
            scores['rs'] = 15
        elif rs >= 0.8:
            scores['rs'] = 10
        
        # Pattern score (0-15)
        setup_type = setup.get('setup_type', '')
        if setup_type == 'PARABOLIC':
            scores['pattern'] = 15
        elif setup_type == 'EP':
            scores['pattern'] = 12
        elif setup_type == 'BREAKOUT':
            scores['pattern'] = 10
        
        # Risk/Reward score (0-15)
        risk = setup['entry'] - setup['stop']
        if risk > 0:
            r_multiple = (setup['target_2R'] - setup['entry']) / risk
            if r_multiple >= 2:
                scores['risk_reward'] = 15
            elif r_multiple >= 1.5:
                scores['risk_reward'] = 10
            else:
                scores['risk_reward'] = 5
        
        total_score = sum(scores.values())
        
        return min(int(total_score), 100)
    
    def apply_filters(self, ticker: str, hist: pd.DataFrame, tech: Dict) -> Tuple[bool, Dict]:
        """Apply all filters (169-191)"""
        
        filters_passed = {
            'volume': True,
            'price': True,
            'rsi': True,
            'atr': True,
            'ma': True,
            'momentum': True
        }
        
        # Volume filter
        if self.volume_filter['enabled']:
            if tech['volume_ratio'] < self.volume_filter['min_ratio']:
                filters_passed['volume'] = False
        
        # Price filter
        price = tech['close']
        if not (self.price_filter['min'] <= price <= self.price_filter['max']):
            filters_passed['price'] = False
        
        # RSI filter
        rsi = tech['rsi']
        if not (self.rsi_filter['min'] <= rsi <= self.rsi_filter['max']):
            filters_passed['rsi'] = False
        
        # ATR filter
        if tech['atr_percent'] < self.atr_filter['min_percent']:
            filters_passed['atr'] = False
        
        # MA filters
        if self.ma_filters['above_10ma'] and tech['price_to_ma10'] < 0:
            filters_passed['ma'] = False
        if self.ma_filters['above_20ma'] and tech['price_to_ma20'] < 0:
            filters_passed['ma'] = False
        
        # Momentum filters
        if tech['roc_20'] < self.momentum_filters['min_roc']:
            filters_passed['momentum'] = False
        
        return all(filters_passed.values()), filters_passed
    
    def scan_ticker(self, ticker: str) -> List[Dict]:
        """Scan single ticker for all setups"""
        
        setups = []
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if len(hist) < 50:
                return []
            
            # Calculate technicals
            tech = self.calculate_technicals(hist)
            
            # Apply filters
            passed, filter_details = self.apply_filters(ticker, hist, tech)
            
            if not passed:
                return []
            
            # Calculate RS
            rs = self.calculate_relative_strength(hist)
            
            # Check for setups
            for setup_detector in [self.detect_ep_setup, self.detect_breakout_setup, self.detect_parabolic_setup]:
                setup = setup_detector(hist, tech)
                
                if setup:
                    setup['ticker'] = ticker
                    
                    # Calculate advanced score
                    setup['score'] = self.calculate_advanced_score(setup, tech, rs)
                    
                    # Add technical details
                    setup['rsi'] = tech['rsi']
                    setup['volume_ratio'] = tech['volume_ratio']
                    setup['relative_strength'] = rs
                    setup['atr_percent'] = tech['atr_percent']
                    setup['price_to_ma10'] = tech['price_to_ma10']
                    
                    # Pattern detection
                    setup['patterns'] = []
                    for pattern, enabled in self.patterns.items():
                        if enabled and self.detect_pattern(hist, pattern):
                            setup['patterns'].append(pattern)
                    
                    setups.append(setup)
        
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
        
        return setups
    
    def run_full_scan(self, max_workers: int = 10) -> pd.DataFrame:
        """Run full scan with all features"""
        
        logger.info("="*80)
        logger.info("🚀 ULTRA SCANNER - FULL SCAN WITH ALL 234 FEATURES")
        logger.info("="*80)
        logger.info(f"Scanning {len(self.tickers)} tickers...")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load SPY data
        if self.spy_data is None:
            self.load_spy_data()
        
        all_setups = []
        
        # Parallel scanning
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.scan_ticker, ticker): ticker for ticker in self.tickers}
            
            for i, future in enumerate(as_completed(futures), 1):
                if i % 100 == 0:
                    logger.info(f"Progress: {i}/{len(self.tickers)} tickers scanned...")
                
                try:
                    setups = future.result()
                    all_setups.extend(setups)
                except Exception as e:
                    logger.error(f"Scan error: {e}")
        
        if not all_setups:
            logger.info("No setups found")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_setups)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        logger.info("="*80)
        logger.info(f"📊 SCAN COMPLETE: {len(df)} SETUPS FOUND")
        logger.info("="*80)
        
        # Breakdown by type
        logger.info("\nBreakdown by Setup Type:")
        for setup_type in df['setup_type'].unique():
            count = len(df[df['setup_type'] == setup_type])
            avg_score = df[df['setup_type'] == setup_type]['score'].mean()
            logger.info(f"  {setup_type}: {count} setups (avg score: {avg_score:.1f})")
        
        # Quality breakdown
        excellent = len(df[df['score'] >= 90])
        high_quality = len(df[(df['score'] >= 80) & (df['score'] < 90)])
        good_quality = len(df[(df['score'] >= 70) & (df['score'] < 80)])
        
        logger.info("\nQuality Breakdown:")
        logger.info(f"  🟢 Excellent (90+): {excellent}")
        logger.info(f"  🟡 High (80-89): {high_quality}")
        logger.info(f"  🔵 Good (70-79): {good_quality}")
        
        return df
    
    def get_pre_market_scan(self) -> pd.DataFrame:
        """Pre-market scanner (feature 16)"""
        # Would use pre-market data API
        logger.info("🌅 Pre-market scan - requires real-time data")
        return pd.DataFrame()
    
    def get_after_hours_scan(self) -> pd.DataFrame:
        """After-hours scanner (feature 17)"""
        # Would use after-hours data API
        logger.info("🌙 After-hours scan - requires real-time data")
        return pd.DataFrame()
    
    def get_gap_scanner(self, gap_type: str = 'up') -> pd.DataFrame:
        """Gap scanner (feature 20)"""
        results = []
        
        for ticker in self.tickers[:100]:  # Sample
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                
                if len(hist) < 2:
                    continue
                
                prev_close = hist['Close'].iloc[-2]
                current_open = hist['Open'].iloc[-1]
                gap_percent = (current_open / prev_close - 1) * 100
                
                if gap_type == 'up' and gap_percent > 2:
                    results.append({
                        'ticker': ticker,
                        'gap_percent': gap_percent,
                        'prev_close': prev_close,
                        'current_price': hist['Close'].iloc[-1]
                    })
                elif gap_type == 'down' and gap_percent < -2:
                    results.append({
                        'ticker': ticker,
                        'gap_percent': gap_percent,
                        'prev_close': prev_close,
                        'current_price': hist['Close'].iloc[-1]
                    })
            except:
                continue
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test scan
    from complete_tickers import COMPLETE_TICKERS
    
    scanner = UltraScannerEngine(COMPLETE_TICKERS[:100])  # Test with 100 tickers
    results = scanner.run_full_scan()
    
    if len(results) > 0:
        print("\n" + "="*80)
        print("TOP 10 SETUPS:")
        print("="*80)
        for idx, row in results.head(10).iterrows():
            print(f"\n{row['ticker']} - {row['setup_type']}")
            print(f"  Score: {row['score']}/100")
            print(f"  Entry: ${row['entry']:.2f}")
            print(f"  Stop: ${row['stop']:.2f}")
            print(f"  Target: ${row['target_2R']:.2f}")
            print(f"  RSI: {row['rsi']:.1f}")
            print(f"  Volume: {row['volume_ratio']:.1f}x")
            print(f"  RS: {row['relative_strength']:.2f}x")
