import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# Custom CSS for background & cards
st.markdown("""
    <style>
        /* Background colors */
        .stApp {
            background-color: #f5f7fa;
        }
        section[data-testid="stSidebar"] {
            background-color: #eaf2f8;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }

        /* KPI card styling */
        .kpi-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin: 10px;
        }
        .predicted {
            background-color: #d6eaf8; /* light blue */
            color: #154360;
        }
        .r2score {
            background-color: #d4efdf; /* light green */
            color: #145a32;
        }
    </style>
""", unsafe_allow_html=True)

  # -------------------- 1. Load Models & Scalers --------------------
@st.cache_resource
def load_resources():
    tsla_model = load_model("tsla_lstm_model.h5", compile=False)
    tsla_scaler = joblib.load("tsla_scaler.joblib")

    googl_model = load_model("googl_lstm_model.h5", compile=False)
    googl_scaler = joblib.load("googl_scaler.joblib")

    return {
        "TSLA": {"model": tsla_model, "scaler": tsla_scaler, "data": "tsla_data.csv"},
        "GOOGL": {"model": googl_model, "scaler": googl_scaler, "data": "googl_data.csv"},
    }

resources = load_resources()

# -------------------- 2. UI Layout --------------------
st.title("📊 Stock Price Prediction Dashboard")
st.markdown("Select a stock and date range to view historical prices and predict the next closing price.")

# Sidebar for user input
st.sidebar.header("User Input")
selected_stock = st.sidebar.selectbox("Select Stock", ["TSLA", "GOOGL"])
start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# -------------------- 3. Load Historical Data --------------------
data_path = resources[selected_stock]["data"]
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Filter by selected dates
df = df.loc[start_date:end_date]

if df.empty:
    st.error("⚠️ No data available for this date range. Please select different dates.")
    st.stop()

# Display historical chart
st.subheader(f"📈 Historical Closing Prices for {selected_stock}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Closing Price"))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# -------------------- 4. Predict Next Day Price --------------------
if st.sidebar.button("Predict Next Day's Price"):

    model = resources[selected_stock]["model"]
    scaler = resources[selected_stock]["scaler"]

    # Last 60 days data
    last_60 = df["Close"].values[-60:]
    scaled = scaler.transform(last_60.reshape(-1, 1))

    X_test = np.array([scaled])
    y_pred = model.predict(X_test)
    predicted_price = scaler.inverse_transform(y_pred)[0][0]

    last_close = df["Close"].iloc[-1]

    # -------------------- 5. KPI Display --------------------
st.success("✅ Prediction complete!")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card predicted">
            Predicted Closing Price<br>
            <span style="font-size:28px;">${predicted_price:.2f}</span><br>
            <span style="font-size:16px;">Change vs Last Close: {((predicted_price - last_close)/last_close)*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card r2score">
            R² Score (Last 60 days)<br>
            <span style="font-size:28px;">{r2:.4f}</span>
        </div>
        """, unsafe_allow_html=True
    )

    # -------------------- 6. Explanation Section --------------------
    st.markdown("""
### ℹ️ How to Interpret the Results
- **Predicted Closing Price** → This is the model’s forecast for the next trading day based on the last 60 days of stock Closing prices.
- **% Change vs Last Close** → Shows how much higher or lower the predicted price is compared to the most recent closing price.
- **R² Score** → A measure of accuracy. Closer to **1.0** means the model explains the stock’s recent movements well.  
  - Example: `0.90` = very strong accuracy, `0.50` = moderate, `0.0` = poor.
""")

    # -------------------- 7. Plot Prediction --------------------
    st.subheader(f"Recent Price vs Predicted Price for {selected_stock}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index[-60:], y=df["Close"].values[-60:], 
                              mode="lines", name="Recent Historical Price"))
    fig2.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], 
                              y=[predicted_price], mode="markers+text", 
                              name="Predicted Price", text=[f"{predicted_price:.2f}"], 
                              textposition="top center", marker=dict(color="red", size=10)))
    fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------- 8. Download Options --------------------
    st.subheader("📥 Download Data")
    # Prepare prediction row
    prediction_df = pd.DataFrame({
        "Date": [df.index[-1] + pd.Timedelta(days=1)],
        "Predicted_Close": [predicted_price]
    }).set_index("Date")

    # Combine with historical data for download
    export_df = pd.concat([df[["Close"]], prediction_df])

    csv = export_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download Historical + Prediction CSV",
        data=csv,
        file_name=f"{selected_stock}_prediction.csv",
        mime="text/csv",
    )

    # -------------------- 9. About Section --------------------
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
