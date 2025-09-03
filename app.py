import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
from datetime import timedelta

# ----------------------------
# Load resources once
# ----------------------------
@st.cache_resource
def load_resources():
    tsla_model = load_model("tsla_lstm_model.h5", compile=False)
    tsla_scaler = joblib.load("tsla_scaler.joblib")
    googl_model = load_model("googl_lstm_model.h5", compile=False)
    googl_scaler = joblib.load("googl_scaler.joblib")
    tsla_data = pd.read_csv("tsla_data.csv", parse_dates=True, index_col=0)
    googl_data = pd.read_csv("googl_data.csv", parse_dates=True, index_col=0)

    return {
        "TSLA": {"model": tsla_model, "scaler": tsla_scaler, "data": tsla_data},
        "GOOGL": {"model": googl_model, "scaler": googl_scaler, "data": googl_data},
    }

resources = load_resources()

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="📊 Stock Price Prediction", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f7fbff; }
    .result-card {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
    }
    .result-card h2 {
        font-size: 32px;
        margin-bottom: 5px;
    }
    .result-card p {
        font-size: 18px;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Stock Price Prediction Dashboard")
st.markdown("Select a stock, date, and prediction horizon. The app will forecast the closing price for the next few days.")

# ----------------------------
# Sidebar inputs
# ----------------------------
stock_choice = st.sidebar.selectbox("📈 Select Stock", ["TSLA", "GOOGL"])
data = resources[stock_choice]["data"]

# Only allow dates with at least 60 prior days
valid_dates = data.index[60:]

# Calendar-style date picker
selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=valid_dates[-1],
    min_value=valid_dates[0],
    max_value=valid_dates[-1]
)
selected_date = pd.to_datetime(selected_date)

# ---- Dynamic horizon control ----
days_remaining = len(data.loc[selected_date:].index) - 1
max_horizon = min(10, days_remaining)

days_ahead = st.sidebar.slider(
    "🔮 Predict how many days ahead?",
    min_value=1,
    max_value=max_horizon,
    value=1
)

# ----------------------------
# Weekend Handling
# ----------------------------
def adjust_to_trading_day(date, direction="forward"):
    """Move date to nearest trading day if it falls on weekend"""
    if date.weekday() == 5:  # Saturday
        return date - timedelta(days=1) if direction == "backward" else date + timedelta(days=2)
    elif date.weekday() == 6:  # Sunday
        return date - timedelta(days=2) if direction == "backward" else date + timedelta(days=1)
    return date

# Adjust actual and prediction dates
actual_date = adjust_to_trading_day(selected_date, direction="backward")
prediction_date = selected_date + timedelta(days=days_ahead)
prediction_date = adjust_to_trading_day(prediction_date, direction="forward")

# Show warnings only if adjustment was needed
if selected_date != actual_date:
    st.warning(f"📌 {selected_date.date()} was a weekend. Showing **last trading day (Friday {actual_date.date()})** for Actual Close.")

if (selected_date + timedelta(days=days_ahead)) != prediction_date and prediction_date.weekday() not in [0]:
    st.warning(f"📌 Prediction landed on weekend. Showing **next trading day (Monday {prediction_date.date()})** instead.")

# ----------------------------
# Prediction
# ----------------------------
idx = data.index.get_loc(actual_date)
scaler = resources[stock_choice]["scaler"]
model = resources[stock_choice]["model"]

past_60 = data["Close"].iloc[idx-60:idx].values.reshape(-1, 1)
scaled_input = scaler.transform(past_60)
last_sequence = scaled_input.copy()

predictions = []
for _ in range(days_ahead):
    pred_scaled = model.predict(np.array([last_sequence]), verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    predictions.append(pred_price)
    last_sequence = np.vstack([last_sequence[1:], pred_scaled])

# ----------------------------
# Results
# ----------------------------
actual_close = data.loc[actual_date, "Close"]

st.markdown(
    f"""
    <div class="result-card">
        <h2>{stock_choice} Forecast</h2>
        <p><b>Selected Date:</b> {selected_date.date()}</p>
        <p><b>Actual Close ({actual_date.date()}):</b> ${actual_close:.2f}</p>
        <p><b>Predicted Close ({prediction_date.date()}):</b> <span style="color:green; font-weight:bold;">${predictions[-1]:.2f}</span></p>
        <p><b>Change vs. Last Close:</b> {((predictions[-1] - actual_close)/actual_close)*100:.2f}%</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[:idx+1], y=data["Close"].iloc[:idx+1],
                         mode="lines", name="Actual Price", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=[prediction_date], y=[predictions[-1]],
                         mode="markers", name="Predicted Price",
                         marker=dict(color="red", size=10)))

fig.update_layout(
    title=f"📉 {stock_choice} Historical vs Forecast ({days_ahead}-Day Horizon)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    plot_bgcolor="#f9fbfd"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Note: Predictions are based on past 60-day patterns. Longer horizons may be less accurate.")

# ----------------------------
# About Section
# ----------------------------
st.markdown("---")
st.markdown("""
## 📘 About this Dashboard

This Stock-Predictor-Interactive-Dashboard predicts **next-day stock closing prices** for **Tesla (TSLA)** and **Google (GOOGL)**.  
It uses a **Long Short-Term Memory (LSTM)** deep learning model, trained on historical stock price data for both Tesla and Google.

### 🔍 How it Works
1. The model takes the **last 60 days of closing prices** as input from the historical saved data.  
2. It learns patterns and trends in stock movements.  
3. It outputs a **forecast for the next trading day’s closing price**.

### ⚠️ Important Notes
- The stock market is highly volatile, influenced by factors like company news, global events, policies, natural calamities, and wars.  
- These cannot be fully captured by predictive models, so predictions are for **educational and research purposes only**.  
- **Not financial advice** — do not use for real trading.  
""")
