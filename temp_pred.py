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

# Load the model from the HDF5 file
def load_lstm_model():
    with h5py.File('temp_model.h5', 'r') as file:
        model_structure = file['model_structure'][()]
        model = model_from_json(model_structure)
        model.load_weights('model_weights.h5')
    return model

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = Client(session=retry_session)

# Function to get coordinates using OpenCage Geocoding API
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
        "current": ["relative_humidity_2m", "precipitation", "cloud_cover", "surface_pressure", "wind_gusts_10m"],
        "timezone": "GMT"
    }

    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

def convert_timestamp_to_local_time(timestamp, time_zone):
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(time_zone))
    return dt_object_local

# Streamlit UI
st.set_page_config(
    page_title="Weather Forecast App",
    page_icon=":partly_sunny:",
    layout="wide",
)

st.title("Today's Weather Forecast")

# Sidebar
st.sidebar.title("Weather Details")
logo_path = "D:/ML and AI/Projects/weather prediction/images.png"
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

        st.write(f"**Coordinates:** {latitude}°N {longitude}°E")
        st.write(f"**Elevation:** {weather_data.Elevation()} m asl")

        current = weather_data.Current()
        timestamp = current.Time()
        dt_local = convert_timestamp_to_local_time(timestamp, 'Asia/Kolkata')

        st.write(f"**Current time (Local):** {dt_local}")
        st.write(f"**Current Humidity:** {current.Variables(0).Value()}")
        st.write(f"**Current Precipitation:** {current.Variables(1).Value()}")
        st.write(f"**Current Cloud Cover:** {current.Variables(2).Value()}")
        st.write(f"**Current Surface Pressure:** {current.Variables(3).Value()}")
        st.write(f"**Current Wind Gusts:** {current.Variables(4).Value()}")

        # Create a new DataFrame in the specified format
        new_df = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'wind_kph': [current.Variables(4).Value()],
            'precip_mm': [current.Variables(1).Value()],
            'pressure_mb': [current.Variables(3).Value()],
            'humidity': [current.Variables(0).Value()],
            'gust_kph': [current.Variables(4).Value()],
            'cloud': [current.Variables(2).Value()]
        })

        # Preprocess the data for LSTM model
        scaler = MinMaxScaler()
        new_df_scaled = scaler.fit_transform(new_df)
        model = load_lstm_model()

        # Reshape the data to match the model's input shape
        new_df_reshaped = new_df_scaled.reshape(new_df_scaled.shape[0], new_df_scaled.shape[1], 1)

        # Make predictions using the LSTM model
        predictions = model.predict(new_df_reshaped)
        temperature_prediction = predictions.flatten()[0]

        st.subheader("Temperature Prediction")
        st.write(f"**Tomorrow's temperature:** {temperature_prediction}°C")

        fig, ax = plt.subplots()
        bar_width = 0.4  # Adjust the width as needed

        # Plot the actual values
        ax.bar(new_df.columns, new_df.values.flatten(), width=bar_width, label='Factors Influencing Temperature')

        # Plot the predicted value with some space from the actual bars
        ax.bar(['Temperature'], [temperature_prediction], width=bar_width, color='red', label='Prediction')

        ax.set_title('Factors Influencing Temperature vs. Predicted Temperature')
        ax.set_xlabel('Weather Variables')
        ax.set_ylabel('Values')
        ax.legend()

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)

    else:
        st.error("Location not found. Please enter a valid location.")
