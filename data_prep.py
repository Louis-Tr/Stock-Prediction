import yfinance as yf
import pandas as pd
import os
import numpy as np
from scipy.signal import argrelextrema


def download_stock_data(companies, start_date, end_date, output_folder):
    """
    Downloads stock price data for a list of companies and saves them in individual CSV files.

    Args:
        companies (list): A list of company tickers.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        output_folder (str): The name of the folder to store the output files.

    Returns:
        None
    """
    # Create the base 'data' directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Create the output folder inside the 'data' directory
    output_path = os.path.join(data_dir, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download data for each company and save as CSV
    for company in companies:
        try:
            # Download data from Yahoo Finance
            data = yf.download(company, start=start_date, end=end_date, interval='1d')
            # Only keep the top-level column names if multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            # Save the data to a CSV file
            csv_file = os.path.join(output_path, f"{company}.csv")
            data.to_csv(csv_file)
            print(f"Data for {company} saved to {csv_file}")
        except Exception as e:
            print(f"Error downloading data for {company}: {e}")

    print(f"All files saved in folder: {output_path}")


def low_high_price_identify(data):
    """
    Identifies the low and high price points in the given data.

    Args:
        data (pd.DataFrame): The stock price data without any modification.

    Returns:
        pd.DataFrame: A DataFrame with columns 'low_points' and 'high_points' indicating the identified points.

    """
    # Ensure the DataFrame has a 'Close' column
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")

    # Identify local maxima (high points) and minima (low points)
    data['Local_Max'] = data['Close'][argrelextrema(data['Close'].values, np.greater, order=5)[0]]
    data['Local_Min'] = data['Close'][argrelextrema(data['Close'].values, np.less, order=5)[0]]

    # Function to enforce strict alternation of extrema
    def enforce_strict_alternation(df, low_col='Local_Min', high_col='Local_Max'):
        extrema_points = []  # List to store the final filtered points

        for i in df.index:
            if not pd.isna(df.at[i, low_col]):  # Low point found
                if not extrema_points or extrema_points[-1][1] == 'high':  # Add if no previous point or last was a high
                    extrema_points.append((i, 'low'))
                elif extrema_points[-1][1] == 'low':  # Replace if consecutive lows, keep the lower
                    if df.at[extrema_points[-1][0], low_col] > df.at[i, low_col]:
                        extrema_points[-1] = (i, 'low')
            elif not pd.isna(df.at[i, high_col]):  # High point found
                if not extrema_points or extrema_points[-1][1] == 'low':  # Add if no previous point or last was a low
                    extrema_points.append((i, 'high'))
                elif extrema_points[-1][1] == 'high':  # Replace if consecutive highs, keep the higher
                    if df.at[extrema_points[-1][0], high_col] < df.at[i, high_col]:
                        extrema_points[-1] = (i, 'high')

        # Split the filtered points into separate columns
        lows_filtered = df.loc[[i for i, t in extrema_points if t == 'low'], ['Close']].rename(
            columns={'Close': 'low_points'})
        highs_filtered = df.loc[[i for i, t in extrema_points if t == 'high'], ['Close']].rename(
            columns={'Close': 'high_points'})

        # Combine the results
        result = pd.DataFrame(index=df.index)
        result['low_points'] = lows_filtered['low_points']
        result['high_points'] = highs_filtered['high_points']
        return result

    # Apply the strict alternation logic and return the result
    return enforce_strict_alternation(data)
