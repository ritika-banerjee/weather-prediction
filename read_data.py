import streamlit as st
import pandas as pd
from openmeteo_requests import Client
from retry_requests import retry
from opencage.geocoder import OpenCageGeocode
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests_cache
import h5py
import base64
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
import openmeteo_requests
import serial

# Function to load the LSTM model
def load_lstm_model():
    with h5py.File('temp_model.h5', 'r') as file:
        model_structure = file['model_structure'][()]
        model = model_from_json(model_structure)
        model.load_weights('model_weights.h5')
    return model

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Function to get weather data using the Open-Meteo API

def get_coordinates(location, api_key):
    geocoder = OpenCageGeocode(api_key)
    result = geocoder.geocode(location)

    if result and len(result):
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    else:
        return None
    
    


def get_weather_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["precipitation", "cloud_cover", "wind_speed_10m", "wind_gusts_10m"],
        "timezone": "GMT"
    }
    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

# Function to preprocess data for the LSTM model
def preprocess(new_df):
    scaler = MinMaxScaler()
    new_df_scaled = scaler.fit_transform(new_df)
    new_df_reshaped = new_df_scaled.reshape(new_df_scaled.shape[0], new_df_scaled.shape[1], 1)
    return new_df_reshaped

def convert_timestamp_to_local_time(timestamp, time_zone):
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(time_zone))
    return dt_object_local

# Streamlit configuration
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon=":partly_sunny:",
    layout="wide",
)

# Load the LSTM model
model = load_lstm_model()

# Streamlit UI
st.title("Weather Prediction and Sensor Integration")

# Sidebar
st.sidebar.title("Weather Details")
logo_path = "D:/ML and AI/Projects/weather prediction/images.png"  # Replace with the actual path
logo_html = f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="logo" width="100%">'
st.sidebar.markdown(logo_html, unsafe_allow_html=True)

# User input for location
location = st.sidebar.text_input("Enter Location:")

if st.sidebar.button("Get Weather Data"):
    opencage_api_key = '0a2224338ab54be5a7a1fe5a7a96ca96'

    coordinates = get_coordinates(location, opencage_api_key)

    if coordinates:
        latitude, longitude = coordinates
        weather_data = get_weather_data(latitude, longitude)

        # Display weather information
        st.subheader("Weather Information")

        st.subheader("Weather Information")

        st.write(f"**Coordinates:** {latitude}°N {longitude}°E")
        st.write(f"**Elevation:** {weather_data.Elevation()} m asl")

        current = weather_data.Current()
        timestamp = current.Time()
        dt_local = convert_timestamp_to_local_time(timestamp, 'Asia/Kolkata')

        current_precipitation = current.Variables(0).Value()
        current_cloud_cover = current.Variables(1).Value()
        current_wind_speed_10m = current.Variables(2).Value()
        current_wind_gusts_10m = current.Variables(3).Value()

        st.write(f"**Current time (Local):** {dt_local}")
        st.write(f"**Current Precipitation:** {current_precipitation}")
        st.write(f"**Current Cloud Cover:** {current_cloud_cover}")
        st.write(f"**Current Wind Gusts:** {current_wind_gusts_10m}")
        st.write(f"**Current Wind Speed:** {current_wind_speed_10m}")

            # Read data from Arduino (simulated for illustration)
        ser = serial.Serial('COM', 9600)
        humidity_data = ser.readline().decode().strip()
        pressure_data = ser.readline().decode().strip()
        
        # Create a new DataFrame with sensor data
        new_df = pd.DataFrame({
            'latitude': [latitude],
                'longitude': [longitude],
                'wind_kph': [current_wind_speed_10m],
                'precip_mm': [current_precipitation],
                'pressure_mb': [pressure_data],
                'humidity': [humidity_data],
                'gust_kph': [current_wind_gusts_10m],
                'cloud': [current_cloud_cover]
            })
        # Preprocess the data for LSTM model
        new_df_reshaped = preprocess(new_df)

        # Make predictions using the LSTM model
        predictions = model.predict(new_df_reshaped)
        temperature_prediction = predictions.flatten()[0]
        st.subheader("Temperature Prediction")
        st.write(f"**Today's temperature:** {temperature_prediction}°C")

        # Plot a graph
        fig, ax = plt.subplots()
        ax.plot(new_df.columns, new_df.values.flatten(), marker='o', label='Actual')
        ax.plot(['Temperature'], [temperature_prediction], marker='o', linestyle='dashed', color='red', label='Prediction')
        ax.set_title('Actual vs. Predicted Temperature')
        ax.set_xlabel('Weather Variables')
        ax.set_ylabel('Values')
        ax.legend()

        st.pyplot(fig)

    else:
        st.error("Location not found. Please enter a valid location.")

