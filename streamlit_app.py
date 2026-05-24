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
stock_name = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

# Download stock data
stock = yf.download(
    stock_name,
    start="2020-01-01",
    end="2025-01-01"
)

# Show dataset
st.subheader("Stock Data")
st.write(stock.head())

# Closing price graph
st.subheader("Closing Price Graph")

fig1 = plt.figure(figsize=(12,6))

plt.plot(stock['Close'])

plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{stock_name} Closing Price")

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

future_price = model.predict(
    [[stock['Close'].iloc[-1]]]
)

st.write(
    "Predicted Future Price:",
    future_price[0]
)
