"""
All trend analysis functions are defined here.

Implementation Logics:
    Extracting different trend feature for machine learning model.
        - Moving Average Convergence Divergence (MACD): A trend-following momentum indicator that shows the relationship
            between two moving averages of a security’s price.
        - Bollinger Bands: A volatility indicator that consists of three lines: the middle line (BBM) is a simple moving
            average, and the upper (BBU) and lower (BBL) bands are standard deviations away from the middle line.
        - Relative Strength Index (RSI): A momentum oscillator that measures the speed and change of price movements.
"""

import pandas as pd

def calculate_macd(data, fast_period, slow_period, signal_period):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - fast_period (int): The short-term EMA period.
    - slow_period (int): The long-term EMA period.
    - signal_period (int): The signal line EMA period.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'MACD', 'Signal', and 'Histogram'.
    """
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal

    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal,
        'Histogram': histogram
    })

def calculate_bollinger_bands(data, period):
    """
    Calculates Bollinger Bands for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the Bollinger Bands.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'BBL', 'BBM', and 'BBU' for the lower, middle, and upper bands.
    """
    sma = data.rolling(window=period).mean()
    std_dev = data.rolling(window=period).std()
    bbu = sma + (2 * std_dev)  # Upper band
    bbl = sma - (2 * std_dev)  # Lower band
    return pd.DataFrame({
        'BBL': bbl,
        'BBM': sma,
        'BBU': bbu
    })

def calculate_rsi(data, period):
    """
    Calculates the Relative Strength Index (RSI) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the RSI.

    Returns:
    - pd.Series: RSI values.
    """
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi