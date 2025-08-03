# idn-stock-forecasting-mlops

Stock Price Forecasting Technical Test
Objective: To evaluate the candidate's proficiency in machine learning model development, MLOps practices (specifically integration with MLflow or Comet), and code quality for a stock price forecasting task.

Scenario: Develop a machine learning model to forecast the closing price of a single Indonesian stock for the next day. The model should demonstrate MLOps principles by integrating with either MLflow or Comet for experiment tracking and model management.

Data Source:

Stock Data: Use `yfinance` to retrieve historical stock data for an Indonesian stock (e.g., UNTR, ASII, BBCA, TLKM). The candidate can choose one.

Requirements:

Data Preparation and Exploration:
Load historical stock data using `yfinance`.
Perform necessary data preprocessing steps (e.g., handling missing values, feature engineering).

Model Development
Develop a machine learning model for time series forecasting using historical stock data. Candidates may choose either of the following modeling approaches based on their preferred strategy:
Important: You must choose only one of the following two approaches either Regression-Based Forecasting or Classification-Based Forecasting. Do not implement both.


Research:
ARIMA/Prophet -> 2-5 years data
LSTM/RNN -> 3-10 years data