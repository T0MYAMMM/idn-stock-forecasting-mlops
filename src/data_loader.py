"""
Load data from yfinance (Indonesian Stock): BBCA, UNTR, ASII, TLKM
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# Load data from yfinance and save to csv (data/raw)
def download_stock_data(ticker: str, start_date:str, end_date:str):
    """
    Note:
    - start_date and end_date must be in the format of 'YYYY-MM-DD'
    - ticker -> stock code (e.g. BBCA, UNTR, ASII, TLKM)ise an error
    """
    try:
        #stock_data = yf.Ticker(ticker)
        #stock_data = stock_data.history(start=start_date, end=end_date)
        print(f"Downloading data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for {ticker} from {start_date} to {end_date}")
        
        os.makedirs('data/raw', exist_ok=True)
        file_path = f'data/raw/{ticker}_{start_date}_to_{end_date}.csv'
        data.to_csv(file_path, index=True)   
        print(f"Data saved to {file_path}")

        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

# Load downloaded data from csv (data/raw)
def load_stock_data_from_csv(file_path:str):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


