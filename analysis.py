import pandas as pd
import pandas_ta as ta


# Moving Averages
def calculate_sma(data, period):
    """
    Calculates the Simple Moving Average (SMA) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the SMA.

    Returns:
    - pd.Series: SMA values.
    """
    return data.ta.sma(length=period)


def calculate_ema(data, period):
    """
    Calculates the Exponential Moving Average (EMA) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the EMA.

    Returns:
    - pd.Series: EMA values.
    """
    return data.ta.ema(length=period)


# Relative Strength Index (RSI)
def calculate_rsi(data, period):
    """
    Calculates the Relative Strength Index (RSI) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the RSI.

    Returns:
    - pd.Series: RSI values.
    """
    return data.ta.rsi(length=period)


# Bollinger Bands
def calculate_bollinger_bands(data, period):
    """
    Calculates Bollinger Bands for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - period (int): The window size for the Bollinger Bands.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'BBL', 'BBM', and 'BBU' for the lower, middle, and upper bands.
    """
    return data.ta.bbands(length=period)


# Moving Average Convergence Divergence (MACD)
def calculate_macd(data, fast_period, slow_period, signal_period):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the given data.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - fast_period (int): The short-term EMA period.
    - slow_period (int): The long-term EMA period.
    - signal_period (int): The signal line EMA period.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'MACD', 'MACDh', and 'MACDs' for the MACD line, histogram, and signal line.
    """
    return data.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period)


# Average Directional Index (ADX)
def calculate_adx(data, period):
    """
    Calculates the Average Directional Index (ADX) for the given data.

    Parameters:
    - data (pd.DataFrame): A DataFrame containing 'high', 'low', and 'close' columns.
    - period (int): The window size for the ADX.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'ADX', '+DI', and '-DI'.
    """
    return data.ta.adx(length=period)


# Parabolic SAR
def calculate_parabolic_sar(data):
    """
    Calculates the Parabolic Stop and Reverse (SAR) for the given data.

    Parameters:
    - data (pd.DataFrame): A DataFrame containing 'high' and 'low' columns.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'PSARl' and 'PSARs' for long and short SAR values.
    """
    return data.ta.psar()


def moving_trend(data, timeframe):
    """
    Calculates the moving trend value based on the specified timeframe.

    Parameters:
    - data (pd.Series): The price data (e.g., 'close' column) as a Pandas Series.
    - timeframe (str): The timeframe to calculate the moving trend.
      Options: 'short', 'medium', or 'long'.

    Returns:
    - pd.Series: The calculated moving trend values.
    """
    if timeframe == 'short':
        # Placeholder for short-term trend calculation
        # Implement custom logic here (e.g., 10-day EMA)
        pass
    elif timeframe == 'medium':
        # Placeholder for medium-term trend calculation
        # Implement custom logic here (e.g., 20-day SMA)
        pass
    elif timeframe == 'long':
        # Placeholder for long-term trend calculation
        # Implement custom logic here (e.g., 50-day SMA)
        pass
    else:
        raise ValueError("Invalid timeframe. Use 'short', 'medium', or 'long'.")
