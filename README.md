📊 Stock Price Prediction Dashboard:
An Interactive web dashboard which uses Long Short-Term Memory (LSTM) deep learning model to predict stock closing prices for historical Tesla (TSLA) and Google (GOOGL) data downloaded from Yahoo Finance from 2010 to 2014 is deployed using Streamlit framework and Hugging face.
The interface is user-friendly ensuring that even non-technical users can easily navigate and use the dashboard.

🚀 Features:
The interactive Dashboard will offer the following options:
1.	Select stock (TSLA-TESLA) or (GOOGL-GOOGLE), choose a date on the calendar (only working days are possible-do not choose Saturdays and Sundays), and select a prediction horizon (1-10 days ahead).
2.	Smart Weekend handling: When user picks on Saturday/Sunday, the app will automatically display the Actual Close for the nearest Friday and Predicted Close of the following Monday. Those predictions that lands on weekends are moved to the following trading day.
3.	Even if the user selects for example Thursday date in calendar and prediction horizon 2 or 3 and lands on say Saturday date and Sunday date respectively the alert message will be generated, and the result card will show coming Monday Prediction Closing price.
4.	Forecast Visualization: Visualize the dashboard using Plotly to show the historical closing prices and prediction markers. The interactive chart enables the users to hover the pointer above data points to view specific price details.
5.	Result Card: Actual Closing price, Predicted Closing price, Change percent vs. previous close, with clear dates.
6.	Educational Section: Interpretation, market volatility notes and disclaimer (Not financial advice).
   
🛠️ Tech Stack:
1.	Frontend & Dashboard: Streamlit framework, Hugging face
2.	Visualization: Plotly
3.	Deep Learning Model: LSTM with TensorFlow/Keras 
4.	Data Handling: Pandas, NumPy
5.	Serialization: Joblib
   
📂 Project Structure
📁 Stock-Prediction-Dashboard
│──src  file 	        		  #Main Source code (Google Colab code)
│── app.py                 	# Main Streamlit app
│── tsla_lstm_model.h5     	# Pre-trained Tesla LSTM model
│── googl_lstm_model.h5     # Pre-trained Google LSTM model
│── tsla_scaler.joblib     	# Fitted scaler for Tesla data
│── googl_scaler.joblib    	# Fitted scaler for Google data
│── tsla_data.csv          	# Tesla stock history (pre-downloaded)
│── googl_data.csv         	# Google stock history (pre-downloaded)
│── requirements.txt       	# Project dependencies
│── README.docx             # Project documentation (this file)

<img width="543" height="263" alt="image" src="https://github.com/user-attachments/assets/af8d7970-95e1-4cef-bbfc-45f493eaebed" />


📖 How to Run Locally
1.	Install the repository: git init
2.	Change to project directory: cd Stock-Prediction-Dashboard
3.	Activate a virtual environment (not compulsory but suggested): python -m venv venv
4.	Activation of the environment 
a.	Mac/Linux: source venv/bin/activate 
b.	Windows: venv\Scripts\activate
5.	Install dependencies: pip install -r requirements.txt
6.	Start the app: streamlit run app.py
7.	Open browser: http://localhost:8501
   
🌐 Deployment
Push all necessary files (models, scalers, data CSVs, app.py, requirements.txt) to Hugging Face Spaces or to Streamlit Cloud.
It does not need an internet connection to run since all the information is downloaded.

⚠️ Disclaimer
This is an educational and research only project.
The movements of stock prices are very volatile and subject to numerous external factors (news, policies, global events, natural calamities, wars, etc.) that that models cannot fully capture.
Not financial advice - not to be used as part of real trading.

👨‍💻 Project Credits
1.	Developer: Kamaxi Patel
2.	Framework: Streamlit
3.	Model: LSTM (TensorFlow/Keras)
4.	Data Source: Pre-downloaded histori
