

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from datetime import timedelta


# Load resources once
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

# Streamlit setup
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

# Sidebar inputs
stock_choice = st.sidebar.selectbox("📈 Select Stock", ["TSLA", "GOOGL"])
data = resources[stock_choice]["data"]

# Only allow dates with at least 60 prior days
valid_dates = data.index[60:]  # skip first 60 days

# Generate list of weekend dates between min and max date
all_dates = pd.date_range(start=valid_dates[0], end=valid_dates[-1], freq='D')
weekends = all_dates[all_dates.weekday >= 5]  # Saturday (5) and Sunday (6)


# Calendar-style date picker
selected_date = st.sidebar.date_input(
    "📅 Select Date",
    value=valid_dates[-1],        # default = latest valid date
    min_value=valid_dates[0],     # earliest selectable date
    max_value=valid_dates[-1],    # latest selectable date
    disabled=weekends.tolist()    # Disable weekends
)
selected_date = pd.to_datetime(selected_date)

# ---- Dynamic horizon control ----
# Days left in dataset after the selected date (counting actual trading days left in your dataset, not calendar gaps.)
days_remaining = len(data.loc[selected_date:].index) - 1

# User can select 1–10 days ahead, but capped at available range
max_horizon = min(10, days_remaining)

days_ahead = st.sidebar.slider(
    "🔮 Predict how many days ahead?",
    min_value=1,
    max_value=max_horizon,
    value=1
)
# Prepare input (last 60 days)
idx = data.index.get_loc(selected_date)
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

# Actual closing price
actual_close = data.loc[selected_date, "Close"]

# Prediction dates
prediction_dates = [selected_date + timedelta(days=i+1) for i in range(days_ahead)]

# Display result card
st.markdown(
   f"""
   <div class="result-card">
   <h2>{stock_choice} Forecast</h2>
   <p><b>Selected Date:</b> {selected_date.date()}</p>
   <p><b>Actual Close:</b> ${actual_close:.2f}</p>
   <p><b>Predicted Close ({days_ahead} day(s) ahead):</b> <span style="color:green; font-weight:bold;">${predictions[-1]:.2f}</span></p>
   <p><b>Change vs. Last Close:</b> {((predictions[-1] - actual_close)/actual_close)*100:.2f}%</p>
   </div>
   """,
   unsafe_allow_html=True
)

       
# Explanation Section
st.markdown("""
### ℹ️ How to Interpret the Results
- **Predicted Closing Price** → This is the model’s forecast for the next trading day based on the last 60 days of stock Closing prices.
- **% Change vs Last Close** → Shows how much higher or lower the predicted price is compared to the most recent closing price.
""")

# Plot with shaded prediction zone
fig = go.Figure()
# Actual price
fig.add_trace(go.Scatter(x=data.index[:idx+1], y=data["Close"].iloc[:idx+1],
              mode="lines", name="Actual Price", line=dict(color="blue")))
# Prediction line
fig.add_trace(go.Scatter(x=prediction_dates, y=predictions,
              mode="lines+markers", name="Predicted Price",
              marker=dict(color="red", size=8)))
# Shade forecast zone
fig.add_vrect(
              x0=prediction_dates[0], x1=prediction_dates[-1],
              fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0
             )

fig.update_layout(
                title=f"📉 {stock_choice} Historical vs Forecast ({days_ahead}-Day Horizon)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                plot_bgcolor="#f9fbfd"
            )
st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Note: Predictions are based on past 60-day patterns. Longer horizons may be less accurate.")


# About Section 
st.markdown("---")  # horizontal separator

st.markdown("""
    ## 📘 About this Dashboard

    This Stock-Predictor-Interactive-Dashboard predicts **next-day stock closing prices** for **Tesla (TSLA)** and **Google (GOOGL)**.
    It uses a **Long Short-Term Memory (LSTM)** deep learning model, trained on historical stock price data for both Tesla and Google.

    ### 🔍 How it Works
    1. The model takes the **last 60 days of closing prices** as input from the Historical saved data
    2. It learns patterns and trends in stock movements .
    3. It outputs a **forecast for the next trading day’s closing price**.

    ### ⚠️ Important Notes
    - The stock market is highly volatile, and its movements depend on many external factors such as company news, global events, government policies,
      natural calamities (e.g., COVID-19, tsunamis, earthquakes), wars, and geopolitical tensions.
    - These factors cannot be fully captured by predictive models like LSTM, and therefore, the predictions should be used for educational and research purposes only, not for financial decisions.
    - **Not financial advice** — do not use for real trading decisions.

    ### 👨‍💻 Project Credits
    - Developed as part of a **Stock Price Prediction System** project.
    - Framework: **Streamlit**
    - Model: **LSTM (Keras/TensorFlow)**
    - Data: Pre-downloaded Tesla and Google historical closing prices from Yahoo Finance
    """)
