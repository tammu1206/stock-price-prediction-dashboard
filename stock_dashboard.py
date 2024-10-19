import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Dashboard title
st.title("Real-Time Stock Price Prediction Dashboard")

# Sidebar for stock selection
stock_ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

# Date range selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Display selected stock ticker
st.write(f"Showing data for: {stock_ticker}")
# Fetch stock data
def load_stock_data(ticker):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Display stock data
stock_data = load_stock_data(stock_ticker)
st.write(stock_data.head())  # Display the first few rows
# Plotting stock prices
st.subheader(f"{stock_ticker} Stock Closing Price Over Time")

def plot_stock_data():
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label="Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price in USD")
    ax.legend()
    st.pyplot(fig)

plot_stock_data()
# Feature and target
stock_data['Day'] = np.arange(1, len(stock_data) + 1)
X = stock_data[['Day']]
y = stock_data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plot predictions vs actual prices
st.subheader("Prediction vs Actual Closing Prices")

def plot_predictions():
    fig, ax = plt.subplots()
    ax.plot(X_test, y_test, 'b', label="Actual Prices")
    ax.plot(X_test, y_pred, 'r', label="Predicted Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price in USD")
    ax.legend()
    st.pyplot(fig)

plot_predictions()
# Future predictions input
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=5)

# Generate future days
future_days = np.arange(len(stock_data) + 1, len(stock_data) + days_to_predict + 1).reshape(-1, 1)

# Predict future prices
future_predictions = model.predict(future_days)

# Display future predictions
st.subheader(f"Predicted Prices for Next {days_to_predict} Days")

def plot_future_predictions():
    fig, ax = plt.subplots()
    ax.plot(future_days, future_predictions, 'g', label="Predicted Future Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price in USD")
    ax.legend()
    st.pyplot(fig)

plot_future_predictions()
