import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys
from io import StringIO
from datetime import datetime, timedelta

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 1️⃣ Load your CSV
df = pd.read_csv("ai_universe_sample_final.csv")
df["ticker"] = df["ticker"].str.strip()

# 2️⃣ Define 6-month window
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# 3️⃣ Download prices with error handling
tickers = df["ticker"].unique().tolist()
data_dict = {}
failed_tickers = []

# Download each ticker individually to handle failures gracefully
for ticker in tickers:
    try:
        # Suppress stderr to avoid yfinance error messages
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        # Use Ticker object for more reliable downloads
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
        
        # Restore stderr
        sys.stderr = old_stderr
        
        # Check if we got valid data
        if hist.empty:
            failed_tickers.append(ticker)
            continue
        
        # Get adjusted close (when auto_adjust=True, Close is already adjusted)
        adj_close = hist['Close']
        
        # Check if we have valid (non-NaN) data
        if isinstance(adj_close, pd.Series) and not adj_close.isna().all() and len(adj_close) > 0:
            data_dict[ticker] = adj_close
        else:
            failed_tickers.append(ticker)
    except Exception as e:
        # Restore stderr in case of exception
        sys.stderr = old_stderr
        failed_tickers.append(ticker)
        continue

if not data_dict:
    raise ValueError("No valid ticker data was downloaded. Please check your internet connection and ticker symbols.")

# Combine all successful downloads into a single DataFrame
data = pd.DataFrame(data_dict)

if failed_tickers:
    print(f"⚠️  Warning: Failed to download data for: {', '.join(failed_tickers)}")
    print(f"   These tickers will have NaN values for metrics.\n")

# 4️⃣ Compute metrics
returns = data.pct_change(fill_method=None).dropna()
expected_return_series = returns.mean() * 252                # annualized mean daily return
volatility_series = returns.std() * np.sqrt(252)             # annualized std
momentum_series = (data.iloc[-1] / data.iloc[0]) - 1        # price change over 6 mo

# Map computed values back to dataframe by ticker
df["expected_return"] = df["ticker"].map(expected_return_series)
df["volatility"] = df["ticker"].map(volatility_series)
df["momentum_6m"] = df["ticker"].map(momentum_series)

# 5️⃣ Save result
df.to_csv("ai_universe_recalculated.csv", index=False)
print("✅ Updated -> ai_universe_recalculated.csv")

