# Qullamaggie Scanner - Streamlit Deployment

## Quick Deploy to Streamlit Cloud

1. **Create a GitHub repository** and upload these files
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Deploy** with these settings:
   - Repository: Your GitHub repo
   - Branch: main
   - Main file: `ultimate_platform_ENHANCED.py`

## Environment Setup

The app will work immediately with default settings. For Telegram notifications:

1. In Streamlit Cloud, go to **Settings → Secrets**
2. Add your secrets in TOML format:

```toml
[telegram]
bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
```

## Files Included

- `ultimate_platform_ENHANCED.py` - Main Streamlit app
- `scanner_qullamaggie_enhanced_complete.py` - Scanner logic
- `complete_tickers.py` - S&P 500 ticker list
- `config.py` - Configuration
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - UI theme configuration

## Features

- ✅ Real-time scanning of S&P 500 stocks
- ✅ Qullamaggie momentum breakout detection
- ✅ Interactive charts with TradingView integration
- ✅ Pre-filtering for efficiency (55x speedup)
- ✅ Top 10% momentum filter
- ✅ Pattern validation (3-10 day consolidation + breakout)
- ✅ Telegram notifications (optional)

## Usage

1. Click **"Scan Now"** to run a scan
2. View results in the table
3. Click any ticker to see detailed charts
4. Optional: Enable auto-refresh for continuous monitoring

## Support

For issues or questions, check the logs in Streamlit Cloud's interface.
