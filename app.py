from flask import Flask, render_template, request
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your saved model and scaler
model = load_model(r'C:\Users\EXCELL  COMPUTERS\Programming_Data_Science\Ml-Projects\Stock_Price\AAPL_stock_price_model.h5')  # Change as needed
scaler = joblib.load(r'C:\Users\EXCELL  COMPUTERS\Programming_Data_Science\Ml-Projects\Stock_Price\AAPL_scaler.pkl')  # Change as needed

# Download stock data
def download_stock_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-10-01")
    return data[['Close']]  # Using the 'Close' column for prediction

# Predict today's stock price
def predict_today_price(model, scaler, ticker, default_time_step=60):
    # Get today's date and the previous day
    today = datetime.today().date()
    start_date = today - timedelta(days=default_time_step)
    
    # Download stock data for the last 'default_time_step' days
    stock_data = yf.download(ticker, start=start_date, end=today)
    
    # Check if stock_data is empty
    if stock_data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")

    # Use the available closing prices
    recent_data = stock_data['Close'].values.reshape(-1, 1)

    # Check the number of available data points
    num_available_data_points = len(recent_data)
    if num_available_data_points < 1:
        raise ValueError("No historical data available for the ticker.")

    # Adjust time_step if not enough data is available
    time_step = min(default_time_step, num_available_data_points)

    # Ensure we have enough data for the model input
    if time_step < default_time_step:
        print(f"Warning: Only {num_available_data_points} data points available; using {time_step} for prediction.")

    # Scale the data
    scaled_recent_data = scaler.transform(recent_data[-time_step:])
    X_input = np.reshape(scaled_recent_data, (1, time_step, 1))

    predicted_price_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        try:
            predicted_price = predict_today_price(model, scaler, ticker)  # Pass the ticker here
            predicted_price = f"${predicted_price:.2f}"  # Format it here
        except ValueError as e:
            error_message = str(e)

    return render_template('index.html', predicted_price=predicted_price, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)