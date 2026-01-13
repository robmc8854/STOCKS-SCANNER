"""
ULTRA TRADING SYSTEM - CONFIGURATION
All settings with Rob's credentials
"""

import os
from datetime import datetime

# =============================================================================
# TELEGRAM CREDENTIALS
# =============================================================================

TELEGRAM_TOKEN = "8337826868:AAGm66OPbebtqQMVHMukf1WN5urNFWUtHWU"
TELEGRAM_CHAT_ID = "688830637"

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================

ACCOUNT_SIZE = 100000  # £100,000
RISK_PER_TRADE = 1.0   # 1% risk per trade
MAX_POSITIONS = 10

# =============================================================================
# SCANNER SETTINGS
# =============================================================================

MIN_SCORE = 70
MIN_ALERT_SCORE = 75

# Auto-scan settings
AUTO_SCAN_ENABLED = True
AUTO_SCAN_INTERVAL = 60  # minutes
POSITION_UPDATE_INTERVAL = 30  # minutes

# Alert thresholds
ALERT_THRESHOLDS = {
    'critical': 90,
    'warning': 80,
    'info': 70,
}

# Quiet hours
QUIET_HOURS = {
    'enabled': True,
    'start': 22,  # 10 PM
    'end': 7,     # 7 AM
}

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

SYSTEM_NAME = "Ultra Trading System"
VERSION = "2.0.0"
TIMEZONE = 'Europe/London'
CURRENCY = 'GBP'
CURRENCY_SYMBOL = '£'

# Directories
LOG_DIR = 'logs'
DATA_DIR = 'data'

# Create directories
for directory in [LOG_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuration class
class Config:
    """Configuration class"""
    
    def __init__(self):
        for key, value in globals().items():
            if key.isupper():
                setattr(self, key, value)

config = Config()
