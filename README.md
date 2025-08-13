# Stock Prediction

## Overview
This repository offers a basic toolkit for exploring stock price trends and building predictive models. Scripts cover data collection, feature engineering, and a skeleton machine‑learning workflow.

## Key Modules
- **data_prep.py** – download historical data from Yahoo Finance and locate alternating local highs and lows in the closing price series.
- **analysis.py** – compute common technical indicators such as moving averages, RSI, Bollinger Bands, MACD, ADX, and Parabolic SAR.
- **trend_analysis.py** – alternate implementations of MACD, Bollinger Bands, and RSI using only pandas operations.
- **model.py** – `StockModel` class that normalizes data, splits training/testing sets, trains a supplied estimator, and evaluates predictions.

## Usage
1. Collect data for specific tickers:
   ```python
   from data_prep import download_stock_data
   download_stock_data(["AAPL"], "2024-01-01", "2024-06-01", "demo")
   ```
2. Load and enrich a DataFrame with indicators from `analysis.py` or `trend_analysis.py`.
3. Train and evaluate a model:
   ```python
   from model import StockModel
   from sklearn.linear_model import LinearRegression

   sm = StockModel()
   sm.fetch_data(df)
   sm.clean_data()
   sm.normalize_data()
   sm.split_data()
   sm.assign_model(LinearRegression())
   sm.train_model()
   metrics = sm.evaluate_model()
   ```

## Notes
- Example plots such as `output.png` and `loess_smoothed_close.png` illustrate data exploration.
- The repository currently includes minimal tests; extend as needed for production use.
