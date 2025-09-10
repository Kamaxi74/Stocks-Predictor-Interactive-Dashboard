import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

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

st.title("📊 Stock Price Prediction Dashboard")
st.markdown("Select a stock, date, and prediction horizon. The app will forecast the closing price for the next few days.")

# ----------------------------
# Sidebar inputs
# ----------------------------
stock_choice = st.sidebar.selectbox("📈 Select Stock", ["TSLA", "GOOGL"])
data = resources[stock_choice]["data"]

# Make sure index is DatetimeIndex and sorted
data = data.sort_index()
data.index = pd.to_datetime(data.index)

# Only allow dates with at least 60 prior trading days
valid_dates = data.index[60:]

selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=valid_dates[-1],
    min_value=valid_dates[0],
    max_value=valid_dates[-1]
)
selected_date = pd.to_datetime(selected_date)

# ----------------------------
# Helper functions
# ----------------------------
def trading_on_or_before(date: pd.Timestamp) -> pd.Timestamp:
    pos = data.index.searchsorted(date, side="right") - 1
    pos = max(0, min(pos, len(data.index) - 1))
    return data.index[pos]

def trading_n_after(date: pd.Timestamp, n: int) -> pd.Timestamp:
    start_idx = data.index.get_loc(date)
    target_idx = min(start_idx + n, len(data.index) - 1)
    return data.index[target_idx]

# Resolve actual trading day
actual_date = trading_on_or_before(selected_date)

# Horizon setup
idx_actual = data.index.get_loc(actual_date)
days_remaining = (len(data.index) - 1) - idx_actual
max_horizon = min(10, days_remaining)

# ----------------------------
# Slider OR no future days
# ----------------------------
if max_horizon == 0:
    days_ahead = 0
    st.sidebar.info("ℹ️ No predictions available beyond this date.")
else:
    days_ahead = st.sidebar.slider(
        "🔮 Predict how many days ahead?",
        min_value=0,
        max_value=max_horizon,
        value=1
    )

# ----------------------------
# Prediction date logic
# ----------------------------
if days_ahead == 0 and selected_date not in data.index:
    # Special case: weekend/holiday with horizon=0 → shift prediction forward
    prediction_date = trading_n_after(actual_date, 1)
    st.warning(
        f"📌 {selected_date.date()} was **not a trading day**. "
        f"Showing last trading day **{actual_date.date()}** for Actual Close. "
        f"Prediction moved to next trading day **{prediction_date.date()}**."
    )
else:
    prediction_date = trading_n_after(actual_date, days_ahead)

# Alert if selected was adjusted
if selected_date != actual_date and not (days_ahead == 0 and selected_date not in data.index):
    st.warning(
        f"📌 {selected_date.date()} was **not a trading day**. "
        f"Showing last trading day **{actual_date.date()}** for Actual Close."
    )

# Alert if prediction had to skip a holiday/weekend
naive_calendar_pred = selected_date + pd.Timedelta(days=days_ahead)
if prediction_date.date() != naive_calendar_pred.date() and days_ahead > 0:
    st.warning(
        f"📌 The prediction date fell on a **non-trading day**. "
        f"Showing next available trading day **{prediction_date.date()}** instead."
    )

# ----------------------------
# Run prediction
# ----------------------------
scaler = resources[stock_choice]["scaler"]
model = resources[stock_choice]["model"]

past_60 = data["Close"].iloc[idx_actual-60:idx_actual].values.reshape(-1, 1)
scaled_input = scaler.transform(past_60)
last_sequence = scaled_input.copy()

predicted_price = None
if days_ahead > 0 or (days_ahead == 0 and selected_date not in data.index):
    steps = days_ahead if days_ahead > 0 else 1
    for _ in range(steps):
        pred_scaled = model.predict(np.array([last_sequence]), verbose=0)
        last_sequence = np.vstack([last_sequence[1:], pred_scaled])
    predicted_price = float(scaler.inverse_transform(pred_scaled)[0][0])
else:
    predicted_price = float(data.loc[actual_date, "Close"])

actual_close = float(data.loc[actual_date, "Close"])

chg_pct = ((predicted_price - actual_close) / actual_close) * 100 if actual_close else 0.0

# ----------------------------
# Result card
# ----------------------------
# Color logic
if predicted_price > actual_close:
    price_color = "green"
    chg_color = "green"
elif predicted_price < actual_close:
    price_color = "red"
    chg_color = "red"
else:
    price_color = "black"
    chg_color = "black"

st.markdown(
    f"""
    <div style="background:#e3f2fd;padding:20px;border-radius:10px;margin-top:10px;text-align:center;">
        <h2>{stock_choice} Forecast</h2>
        <p><b>Selected Date:</b> {selected_date.date()}</p>
        <p><b>Actual Close ({actual_date.date()}):</b> ${actual_close:.2f}</p>
        <p><b>Predicted Close ({prediction_date.date()}):</b>
            <span style="color:{price_color}; font-weight:bold;">${predicted_price:.2f}</span>
        </p>
        <p><b>Change vs. Last Close:</b>
            <span style="color:{chg_color}; font-weight:bold;">{chg_pct:.2f}%</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.index[:idx_actual+1],
    y=data["Close"].iloc[:idx_actual+1],
    mode="lines",
    name="Actual Price",
    line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=[prediction_date],
    y=[predicted_price],
    mode="markers+text",
    name="Predicted Price",
    marker=dict(size=10),
    text=[f"{predicted_price:.2f}"],
    textposition="top center"
))
fig.update_layout(
    title=f"📉 {stock_choice} Historical vs Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    plot_bgcolor="#f9fbfd"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Note: Predictions use the last 60 trading days. Longer horizons may be less accurate.")

# ----------------------------
# About Section
# ----------------------------
st.markdown("---")
st.markdown("""
## 📘 About this Dashboard
This Stock-Predictor-Interactive-Dashboard predicts **next-day stock closing prices** for **Tesla (TSLA)** and **Google (GOOGL)**.
It uses a **Long Short-Term Memory (LSTM)** deep learning model, trained on historical stock price data.

### 🔍 How it Works
1. The model takes the **last 60 trading days of closing prices** as input.
2. It learns patterns and trends in stock movements.
3. It outputs a **forecast for the next trading day’s closing price** (or `n` trading days ahead).

### ⚠️ Important Notes
- Markets are volatile; predictions are for **educational purposes only** and **not financial advice**.
- Holidays and weekends are handled automatically using the trading days present in the CSVs.
""")
