# preprocessing.py

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Data/yellow_tripdata_2015-01.csv')

# Convert datetime columns to datetime type
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Add hour and day of the week features
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

# Filter out any trips with non-positive fare amounts or distances
df = df[(df['total_amount'] > 0) & (df['trip_distance'] > 0)]

# Drop unnecessary columns
df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag'], axis=1)

# Handle missing values by dropping rows with missing data
df = df.dropna()

# Save the preprocessed data to a new CSV file
df.to_csv('nyc_taxi_preprocessed_data.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to 'nyc_taxi_preprocessed_data.csv'.")
