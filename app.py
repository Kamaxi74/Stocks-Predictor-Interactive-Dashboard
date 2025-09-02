import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.graph_objs as go

# ----------------- Load resources once -----------------
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

# ----------------- Streamlit setup -----------------
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

# ----------------- Sidebar inputs -----------------
stock_choice = st.sidebar.selectbox("📈 Select Stock", ["TSLA", "GOOGL"])
data = resources[stock_choice]["data"]

# Only allow dates with at least 60 prior days
valid_dates = data.index[60:]

selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=valid_dates[-1],
    min_value=valid_dates[0],
    max_value=valid_dates[-1]
)
selected_date = pd.to_datetime(selected_date)

# ---- Weekend smart handling ----
# If Saturday → shift actual date to Friday, prediction date to Monday
# If Sunday   → shift actual date to Friday, prediction date to Monday
if selected_date.weekday() == 5:  # Saturday
    actual_date = selected_date - timedelta(days=1)
    st.warning("📌 Selected date is Saturday. Using Friday's actual close and forecasting Monday's price.")
elif selected_date.weekday() == 6:  # Sunday
    actual_date = selected_date - timedelta(days=2)
    st.warning("📌 Selected date is Sunday. Using Friday's actual close and forecasting Monday's price.")
else:
    actual_date = selected_date

# Remaining days in dataset after the selected actual date
days_remaining = len(data.loc[actual_date:].index) - 1
max_horizon = min(10, days_remaining)

# Horizon slider
days_ahead = st.sidebar.slider(
    "🔮 Predict how many days ahead?",
    min_value=1,
    max_value=max_horizon,
    value=1
)

# ----------------- Prediction Logic -----------------
if actual_date not in data.index:
    st.error("Date not found in dataset. Please select another date.")
else:
    idx = data.index.get_loc(actual_date)
    if idx < 60:
        st.error("Not enough prior data to make prediction (need 60 days).")
    else:
        scaler = resources[stock_choice]["scaler"]
        model = resources[stock_choice]["model"]

        # Prepare last 60 days for input
        past_60 = data["Close"].iloc[idx-60:idx].values.reshape(-1, 1)
        scaled_input = scaler.transform(past_60)
        last_sequence = scaled_input.copy()

        predictions = []
        for _ in range(days_ahead):
            pred_scaled = model.predict(np.array([last_sequence]), verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            last_sequence = np.vstack([last_sequence[1:], pred_scaled])

        # Actual & predicted dates
        actual_close = data.loc[actual_date, "Close"]
        # Prediction dates (with weekend handling + warnings)
        raw_prediction_dates = [selected_date + timedelta(days=i+1) for i in range(days_ahead)]
        prediction_dates = []

        for d in raw_prediction_dates:
            if d.weekday() == 5:  # Saturday
               shifted = d + timedelta(days=2)
               prediction_dates.append(shifted)
               st.warning(f"⚠️ Prediction date {d.date()} fell on Saturday. Shifted to Monday ({shifted.date()}).")
            elif d.weekday() == 6:  # Sunday
               shifted = d + timedelta(days=1)
               prediction_dates.append(shifted)
               st.warning(f"⚠️ Prediction date {d.date()} fell on Sunday. Shifted to Monday ({shifted.date()}).")
            else:
               # If already Monday–Friday, no warning
               prediction_dates.append(d)
            
        # Adjust if prediction lands on Sat/Sun → push to Monday
        adjusted_prediction_dates = []
        for d in prediction_dates:
            if d.weekday() == 5:  # Sat → shift to Mon
                d = d + timedelta(days=2)
            elif d.weekday() == 6:  # Sun → shift to Mon
                d = d + timedelta(days=1)
            adjusted_prediction_dates.append(d)

        # ----------------- Result Card -----------------
        st.markdown(
            f"""
            <div class="result-card">
                <h2>{stock_choice} Forecast</h2>
                <p><b>Actual Close Date:</b> {actual_date.date()}</p>
                <p><b>Actual Close Price:</b> ${actual_close:.2f}</p>
                <p><b>Prediction Date ({days_ahead} day(s) ahead):</b> {adjusted_prediction_dates[-1].date()}</p>
                <p><b>Predicted Close Price:</b>
                   <span style="color:green; font-weight:bold;">${predictions[-1]:.2f}</span></p>
                <p><b>Change vs. Last Close:</b> {((predictions[-1] - actual_close)/actual_close)*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------- Plot -----------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[:idx+1], y=data["Close"].iloc[:idx+1],
                                 mode="lines", name="Actual Price", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=adjusted_prediction_dates, y=predictions,
                                 mode="lines+markers", name="Predicted Price",
                                 marker=dict(color="red", size=8)))
        fig.add_vrect(
            x0=adjusted_prediction_dates[0], x1=adjusted_prediction_dates[-1],
            fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0
        )

        fig.update_layout(
            title=f"📉 {stock_choice} Historical vs Forecast ({days_ahead}-Day Horizon)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            plot_bgcolor="#f9fbfd"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ Note: Predictions are adjusted to skip weekends. If forecast lands on Saturday or Sunday, results are shifted to Monday.")

# ----------------- About Section -----------------
st.markdown("---")

st.markdown("""
## 📘 About this Dashboard

This Stock-Predictor-Interactive-Dashboard predicts **stock closing prices** for **Tesla (TSLA)** and **Google (GOOGL)**.
It uses a **Long Short-Term Memory (LSTM)** deep learning model, trained on historical stock price data.

### 🔍 How it Works
1. The model takes the **last 60 days of closing prices**.
2. It learns patterns in stock movements.
3. It forecasts the **next trading day’s closing price**.

### ⚠️ Important Notes
- Stock markets are **closed on Saturdays and Sundays**. This dashboard automatically adjusts predictions to skip weekends.
- The stock market is highly volatile, and its movements depend on many external factors such as company news, global events, 
  government policies, natural calamities (e.g., COVID-19, tsunamis, earthquakes), wars, and geopolitical tensions.
- Use this tool for **educational and research purposes only**. Not financial advice.

### 👨‍💻 Project Credits
- Developed as part of a **Stock Price Prediction System** project.
- Framework: **Streamlit**
- Model: **LSTM (Keras/TensorFlow)**
- Data: Pre-downloaded Tesla and Google stock data from Yahoo Finance from 2010 to 2024
""")
