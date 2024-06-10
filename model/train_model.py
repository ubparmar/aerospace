import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import re
import lightgbm as lgb

# Load the data
file_path = 'data/Final.csv'
data = pd.read_csv(file_path)

# Function to handle arrival times with +2
def parse_arrival_time(time_str):
    if '+' in time_str:
        return time_str.split('+')[0]
    return time_str

data['Arrival_Clean'] = data['Arrival'].apply(parse_arrival_time)

# Extract features from 'Departure' and 'Arrival_Clean'
data['Departure_Hour'] = pd.to_datetime(data['Departure'], format='%I:%M %p').dt.hour
data['Departure_Minute'] = pd.to_datetime(data['Departure'], format='%I:%M %p').dt.minute
data['Arrival_Hour'] = pd.to_datetime(data['Arrival_Clean'], format='%I:%M %p').dt.hour
data['Arrival_Minute'] = pd.to_datetime(data['Arrival_Clean'], format='%I:%M %p').dt.minute

# Extract date features
data['Day'] = pd.to_datetime(data['Date']).dt.day
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Year'] = pd.to_datetime(data['Date']).dt.year

# Drop original date and time columns
data = data.drop(columns=['Departure', 'Arrival', 'Arrival_Clean', 'Date', 'date'])

# Function to convert duration strings to minutes
def duration_to_minutes(duration):
    match = re.match(r'((?P<hours>\d+)h )?(?P<minutes>\d+)m', duration)
    if not match:
        return 0
    parts = match.groupdict()
    return int(parts['hours'] or 0) * 60 + int(parts['minutes'] or 0)

# Apply the conversion to stopover times
stopover_cols = ['Stopover_1_Time', 'Stopover_2_Time', 'Stopover_3_Time', 'Total_Stopover_Time']
for col in stopover_cols:
    data[col] = data[col].fillna('0').apply(duration_to_minutes)

# Fill missing values for stopover airports with 'None'
stopover_airport_cols = ['Stopover_1_Airport', 'Stopover_2_Airport', 'Stopover_3_Airport']
for col in stopover_airport_cols:
    data[col] = data[col].fillna('None')

# Fill missing values in 'Operated' with 'None'
data['Operated'] = data['Operated'].fillna('None')

# Drop unnecessary columns
unnecessary_columns = ['Stopover_1_Time', 'Stopover_1_Airport', 'Stopover_2_Time', 'Stopover_2_Airport', 'Stopover_3_Time', 'Stopover_3_Airport', 'Operated']
data = data.drop(columns=unnecessary_columns)

# Encode categorical variables
categorical_columns = ['Airline', 'Source', 'Destination', 'Number of Stops', 'Class']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Separate features and target variable
X = data.drop(columns=['price in CAD'])
y = data['price in CAD']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM Regressor
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the model
joblib.dump(model, 'model/airline_price_model.pkl')

# Save the label encoders
for col, le in encoders.items():
    joblib.dump(le, f'model/{col}_encoder.pkl')
