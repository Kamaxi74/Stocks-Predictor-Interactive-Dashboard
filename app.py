import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta

#-----1. Load Pre-trained Models and Scalers----
#Use Streamlit's caching to load model only once
@st.cache_resource
def load_resources():
    try:
        tsla_model = load_model('tsla_lstm_model.h5', compile=False)
        tsla_scaler = joblib.load('tsla_scaler.joblib')
        googl_model = load_model('googl_lstm_model.h5', compile=False)
        googl_scaler = joblib.load('googl_scaler.joblib')
        return{
            'TSLA' :{'model' :tsla_model, 'scaler':tsla_scaler},
            'GOOGL':{'model':googl_model, 'scaler':googl_scaler}
        }
    except Exception as e:
        st.error(f"Error loading model :{e}. Please ensure you've saved them correctly")
        return None

models_and_scalers = load_resources()

#-----2.Dashboard UI and Prediction Logic---------
st.set_page_config(page_title='Stock Price Predictor', layout='wide')
st.title('Stock Price Prediction Dashboard')
st.markdown("Predict the next day's closing price")

if models_and_scalers:
   #Sidebar for user Input
   st.sidebar.header('User Input')
   selected_ticker = st.sidebar.selectbox(
            'Select a stock ticker' ,
             ('TSLA' , 'GOOGL')
             )

   #Prediction button
   if st.sidebar.button("Predict Next Day's Price"):
    with st.spinner(f'Fetching data and predicting for {selected_ticker}...'):
        try:
            # Get correct model and scaler
            model = models_and_scalers[selected_ticker]['model']
            scaler = models_and_scalers[selected_ticker]['scaler']

            # Load pre-downloaded CSV data
            if selected_ticker == "TSLA":
                data = pd.read_csv("tsla_data.csv", parse_dates=["Date"], index_col="Date")
            elif selected_ticker == "GOOGL":
                data = pd.read_csv("googl_data.csv", parse_dates=["Date"], index_col="Date")

            # Ensure we have enough rows
            if data.empty or len(data['Close']) < 60:
                st.error(f"Not enough stock data available for {selected_ticker}. "
                         f"Got only {len(data)} rows, need at least 60.")
            else:
                # Take last 60 days closing prices
                last_60_days = data['Close'].iloc[-60:].values.reshape(-1, 1)

                # Scale & reshape
                scaled_input = scaler.transform(last_60_days)
                X_input = np.reshape(scaled_input, (1, 60, 1))

                # Make prediction
                predicted_price_scaled = model.predict(X_input)
                predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

                st.success("Prediction complete!")

                # Display result
                st.metric(
                    label=f"Predicted Closing Price for {selected_ticker}",
                    value=f"${predicted_price:.2f}"
                )

                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index[-60:],
                    y=data['Close'].iloc[-60:],
                    mode='lines',
                    name='Recent Historical Price'
                ))
                fig.add_trace(go.Scatter(
                    x=[data.index[-1] + timedelta(days=1)],
                    y=[predicted_price],
                    mode='markers+text',
                    name='Predicted Price',
                    marker=dict(size=10, color="red"),
                    text=[f"{predicted_price:.2f}"],
                    textposition="top right"
                ))
                fig.update_layout(
                    title=f"Recent Price vs. Predicted Price for {selected_ticker}",
                    xaxis_title="Date",
                    yaxis_title="Stock Price",
                    template="plotly_dark",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")



