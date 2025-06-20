import yfinance as yf
import pandas as pd


def validate_tickers(tickers):
    """
    Validates tickers by checking if they return any historical data.
    
    Args:
        tickers (list): Ticker symbols

    Returns:
        list: Valid tickers
    """
    valid = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                valid.append(ticker)
        except Exception:
            continue
    return valid



def format_weights(weights_dict):
    """
    Converts weight dict to sorted DataFrame.
    
    Args:
        weights_dict (dict): Ticker: weight pairs

    Returns:
        pd.DataFrame: Sorted weight DataFrame
    """
    df = pd.DataFrame(list(weights_dict.items()), columns=["Ticker", "Weight"])
    df = df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
    return df

