import streamlit as st
import pandas as pd
import joblib
import re
import datetime
from pathlib import Path
import base64

# Define the base path to the model directory
base_path = Path(__file__).parent

# Load the model and encoders
model_path = base_path / 'model' / 'airline_price_model.pkl'
model = joblib.load(model_path)

encoders = {}
for col in ['Airline', 'Source', 'Destination', 'Number of Stops', 'Class']:
    encoder_path = base_path / 'model' / f'{col}_encoder.pkl'
    encoders[col] = joblib.load(encoder_path)

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the background image file
bg_img_path = base_path / 'image' / 'airplane.jpg'
bg_img_base64 = get_base64_of_bin_file(bg_img_path)

# Custom CSS for background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_img_base64}");
        background-size: cover;
    }}
    .stHorizontal {{
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    .stHorizontal > div {{
        margin-right: 20px;
    }}
    .profile-pic {{
        width: 150px;
        border-radius: 50%;
        transition: opacity 0.5s ease-in-out;
        opacity: 0.8;
    }}
    .profile-pic:hover {{
        opacity: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display profile picture with fading effect
st.image(f"data:image/jpg;base64,{bg_img_base64}", use_column_width=True, caption="Airline Ticket Price Prediction")

# User inputs
st.title('Airline Ticket Price Prediction')

st.subheader('Airline Information')
with st.form(key='airline_form'):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        airline = st.selectbox('Airline', encoders['Airline'].classes_)
    with col2:
        source = st.selectbox('Source', encoders['Source'].classes_)
    with col3:
        destination = st.selectbox('Destination', encoders['Destination'].classes_)
    st.form_submit_button(label='Submit')

st.subheader('Flight Details')
with st.form(key='flight_form'):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        number_of_stops = st.selectbox('Number of Stops', encoders['Number of Stops'].classes_)
    with col2:
        flight_class = st.selectbox('Class', encoders['Class'].classes_)
    with col3:
        departure_time = st.time_input('Departure Time', value=datetime.time(10, 0))
        arrival_time = st.time_input('Arrival Time', value=datetime.time(10, 0))
    st.form_submit_button(label='Submit')

st.subheader('Date and Additional Information')
with st.form(key='date_form'):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        flight_date = st.date_input('Flight Date')
    with col2:
        days_left = st.slider('Days Left for Flight', 0, 365, 30)
    with col3:
        total_stopover_time = st.text_input('Total Stopover Time (e.g., "3h 25m")', '0')
    st.form_submit_button(label='Submit')

# Function to convert duration strings to minutes
def duration_to_minutes(duration):
    match = re.match(r'((?P<hours>\d+)h )?(?P<minutes>\d+)m', duration)
    if not match:
        return 0
    parts = match.groupdict()
    return int(parts['hours'] or 0) * 60 + int(parts['minutes'] or 0)

# Extract day, month, and year from the selected date
day = flight_date.day
month = flight_date.month
year = flight_date.year

# Encode user inputs
encoded_inputs = [
    encoders['Airline'].transform([airline])[0],
    encoders['Source'].transform([source])[0],
    encoders['Destination'].transform([destination])[0],
    encoders['Number of Stops'].transform([number_of_stops])[0],
    encoders['Class'].transform([flight_class])[0],
    departure_time.hour,
    departure_time.minute,
    arrival_time.hour,
    arrival_time.minute,
    day,
    month,
    year,
    duration_to_minutes(total_stopover_time),
    days_left
]

# Predict price
if st.button('Predict Price'):
    prediction = model.predict([encoded_inputs])
    st.write(f'Predicted Price: CAD {prediction[0]:.2f}')
