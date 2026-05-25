from streamlit_autorefresh import st_autorefresh
import mplfinance as mpf
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("📈 Stock Price Prediction App")

st.markdown("""
### Real-Time Stock Market Prediction Dashboard  
Built using Machine Learning, Streamlit, and Financial Data Analysis
""")

# ---------------- AUTO REFRESH ---------------- #

st_autorefresh(
    interval=60000,
    key="stock_refresh"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Dashboard Settings")

stock_name = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

stock_name = stock_name.upper()

# ---------------- DOWNLOAD STOCK DATA ---------------- #

stock = yf.download(
    stock_name,
    start="2020-01-01"
)

# ---------------- FIX MULTI-INDEX COLUMNS ---------------- #

if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

# ---------------- STOCK NEWS ---------------- #

st.divider()

st.subheader(f"📰 Latest {stock_name} News")

try:

    ticker = yf.Ticker(stock_name)

    news = ticker.news

    if news:

        for item in news[:5]:
            title = item.get("title")
            
            if not title:
                continue

            publisher = item.get(
                "publisher",
                "Unknown Publisher"
            )

            link = item.get("link", "")

            st.markdown(f"### {title}")

            st.write(publisher)

            st.write(link)

            st.divider()

    else:

        st.write("No news available.")

except Exception:

    st.warning("Unable to load news at the moment.")

# ---------------- LATEST PRICE ---------------- #

latest_close = stock['Close'].to_numpy().flatten()[-1]

previous_close = stock['Close'].to_numpy().flatten()[-2]

change = latest_close - previous_close

change_percent = (change / previous_close) * 100

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label=f"{stock_name} Latest Price",
        value=f"${latest_close:.2f}"
    )

with col2:

    st.metric(
        label="Daily Change",
        value=f"{change:.2f} USD",
        delta=f"{change_percent:.2f}%"
    )

# ---------------- MOVING AVERAGES ---------------- #

stock['MA20'] = stock['Close'].rolling(20).mean()

stock['MA50'] = stock['Close'].rolling(50).mean()

# Buy/Sell signals

stock['Signal'] = 0

stock.loc[
    stock['MA20'] > stock['MA50'],
    'Signal'
] = 1

stock.loc[
    stock['MA20'] < stock['MA50'],
    'Signal'
] = -1

# ---------------- CANDLESTICK CHART ---------------- #

st.divider()

st.subheader("Candlestick Chart")

candlestick_data = stock[
    ['Open', 'High', 'Low', 'Close', 'Volume']
].copy()

# Convert columns to numeric

for column in candlestick_data.columns:

    candlestick_data[column] = pd.to_numeric(
        candlestick_data[column],
        errors='coerce'
    )

candlestick_data = candlestick_data.dropna()

fig2, axlist = mpf.plot(
    candlestick_data,
    type='candle',
    mav=(20, 50),
    volume=True,
    style='yahoo',
    figsize=(12, 6),
    warn_too_much_data=10000,
    returnfig=True
)

st.pyplot(fig2)

# ---------------- TRADING SIGNALS ---------------- #

st.divider()

st.subheader("📈 Buy/Sell Trading Signals")

latest_signal = stock['Signal'].iloc[-1]

if latest_signal == 1:

    st.success("BUY Signal Detected")

elif latest_signal == -1:

    st.error("SELL Signal Detected")

else:

    st.warning("No Clear Signal")

# ---------------- STOCK DATA TABLE ---------------- #

st.divider()

st.subheader("Stock Data")

st.write(stock.head())

# ---------------- CLOSING PRICE GRAPH ---------------- #

st.divider()

st.subheader("Closing Price Graph with Signals")

fig1 = plt.figure(figsize=(12, 6))

plt.plot(
    stock['Close'],
    label='Close Price'
)

plt.plot(
    stock['MA20'],
    label='20-Day MA'
)

plt.plot(
    stock['MA50'],
    label='50-Day MA'
)

# Buy signals

buy_signals = stock[
    stock['Signal'] == 1
]

plt.scatter(
    buy_signals.index,
    buy_signals['Close'],
    marker='^',
    s=100,
    label='BUY Signal'
)

# Sell signals

sell_signals = stock[
    stock['Signal'] == -1
]

plt.scatter(
    sell_signals.index,
    sell_signals['Close'],
    marker='v',
    s=100,
    label='SELL Signal'
)

plt.xlabel("Date")

plt.ylabel("Price")

plt.title(f"{stock_name} Trading Signals")

plt.legend()

st.pyplot(fig1)

# ---------------- MACHINE LEARNING ---------------- #

future_days = 30

stock['Prediction'] = stock['Close'].shift(-future_days)

stock = stock.dropna()

X = np.array(stock[['Close']])

y = np.array(stock['Prediction'])

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ---------------- #

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ---------------- #

predictions = model.predict(X_test)

# ---------------- ACCURACY ---------------- #

accuracy = model.score(X_test, y_test)

st.divider()

st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy:.2f}")

# ---------------- ACTUAL VS PREDICTED ---------------- #

st.divider()

st.subheader("Actual vs Predicted Prices")

fig3 = plt.figure(figsize=(12, 6))

plt.plot(
    y_test[:100],
    label='Actual'
)

plt.plot(
    predictions[:100],
    label='Predicted'
)

plt.xlabel("Data Points")

plt.ylabel("Price")

plt.legend()

st.pyplot(fig3)

# ---------------- FUTURE PREDICTION ---------------- #

st.divider()

st.subheader("Future Stock Price Prediction")

latest_price = stock['Close'].to_numpy().flatten()[-1]

future_price = model.predict(
    [[latest_price]]
)

st.write(
    f"Predicted Price After {future_days} Days:",
    round(float(future_price[0]), 2)
)