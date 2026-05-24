import mplfinance as mpf
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("📈 Stock Price Prediction App")

# User input
stock_name = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

# Convert to uppercase
stock_name = stock_name.upper() 

# Show selected stock
st.write("Selected Stock:", stock_name)

# Download stock data
stock_name = stock_name.upper()

stock = yf.download(
    stock_name,
    start="2020-01-01",
    end="2025-01-01"
)

# Moving averages

stock['MA20'] = stock['Close'].rolling(20).mean()

stock['MA50'] = stock['Close'].rolling(50).mean()

# Candlestick chart

st.subheader("Candlestick Chart")

# Create clean dataframe
candlestick_data = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Flatten columns if needed
candlestick_data.columns = [
    col[0] if isinstance(col, tuple) else col
    for col in candlestick_data.columns
]

# Convert all columns to numeric
for column in candlestick_data.columns:
    candlestick_data[column] = pd.to_numeric(
        candlestick_data[column],
        errors='coerce'
    )

# Remove missing values
candlestick_data = candlestick_data.dropna()

# Create candlestick chart
fig2, axlist = mpf.plot(
    candlestick_data,
    type='candle',
    mav=(20, 50),
    volume=True,
    style='yahoo',
    figsize=(12,6),
    warn_too_much_data=10000,
    returnfig=True
)

st.pyplot(fig2)

# Show dataset
st.subheader("Stock Data")
st.write(stock.head())

# Closing price graph
st.subheader("Closing Price Graph")

fig1 = plt.figure(figsize=(12,6))

plt.plot(stock['Close'], label='Close Price')

plt.plot(stock['MA20'], label='20-Day MA')

plt.plot(stock['MA50'], label='50-Day MA')

plt.xlabel("Date")
plt.ylabel("Price")

plt.title(f"{stock_name} Stock Analysis")

plt.legend()

st.pyplot(fig1)

# Create prediction column
future_days = 30

stock['Prediction'] = stock[['Close']].shift(-future_days)

# Remove missing values
stock = stock.dropna()

# Features and labels
X = np.array(stock[['Close']])

y = np.array(stock['Prediction'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)

st.subheader("Model Accuracy")

st.write(accuracy)

# Actual vs Predicted graph
st.subheader("Actual vs Predicted Prices")

fig2 = plt.figure(figsize=(12,6))

plt.plot(y_test[:100], label='Actual')

plt.plot(predictions[:100], label='Predicted')

plt.xlabel("Data Points")
plt.ylabel("Price")

plt.legend()

st.pyplot(fig2)

# Future prediction
st.subheader("Future Stock Price Prediction")

latest_price = stock['Close'].to_numpy().flatten()[-1]

future_price = model.predict([[latest_price]])

st.write(
    "Predicted Future Price:",
    future_price[0]
)