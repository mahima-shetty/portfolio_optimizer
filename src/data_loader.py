import yfinance as yf
import pandas as pd

def download_price_data(tickers, start_date="2018-01-01", end_date=None, interval="1d"):
    """
    Downloads historical adjusted close prices for a list of tickers from Yahoo Finance.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str or None): End date for data (default: today)
        interval (str): Data interval (e.g., '1d', '1wk', '1mo')

    Returns:
        pd.DataFrame: Adjusted close price data
    """
    if not tickers:
        raise ValueError("Ticker list is empty.")

    print(f"[INFO] Downloading data for: {tickers}")
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}")
        return pd.DataFrame()

    # Extract Adjusted Close
    if len(tickers) == 1:
        # Single ticker case
        adj_close = data['Adj Close'].to_frame(name=tickers[0])
    else:
        adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})

    adj_close.dropna(axis=0, how='all', inplace=True)
    return adj_close
