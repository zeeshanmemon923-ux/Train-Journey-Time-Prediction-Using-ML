import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

print("Loading dataset...")

# Load dataset
df = pd.read_csv("Dataset1.csv")

# Remove duplicates
df = df.drop_duplicates()

# Replace text placeholders in time columns
df["Arrival_time"] = df["Arrival_time"].replace(["Source"], np.nan)
df["Departure_Time"] = df["Departure_Time"].replace(["Destination"], np.nan)

# Convert to datetime
df["Arrival_time"] = pd.to_datetime(df["Arrival_time"], errors="coerce")
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], errors="coerce")

# Drop rows where time conversion failed
df = df.dropna(subset=["Arrival_time", "Departure_Time"])

# Convert time to minutes
df["Arrival_Minutes"] = df["Arrival_time"].dt.hour * 60 + df["Arrival_time"].dt.minute
df["Departure_Minutes"] = df["Departure_Time"].dt.hour * 60 + df["Departure_Time"].dt.minute

# Sort properly
df = df.sort_values(["Train_No", "Distance"])

print("Creating journey duration feature...")

# Get start and end times
journey_times = df.groupby("Train_No").agg(
    Start_Time=("Departure_Minutes", "first"),
    End_Time=("Arrival_Minutes", "last")
).reset_index()

# Calculate duration
journey_times["Journey_Duration_Minutes"] = (
    journey_times["End_Time"] - journey_times["Start_Time"]
)

# Fix overnight journeys
journey_times.loc[
    journey_times["Journey_Duration_Minutes"] < 0,
    "Journey_Duration_Minutes"
] += 24 * 60

# Feature engineering
total_distance = df.groupby("Train_No")["Distance"].max().reset_index()
total_stops = df.groupby("Train_No").size().reset_index(name="Total_Stops")

# Merge features
final_dataset = journey_times.merge(total_distance, on="Train_No")
final_dataset = final_dataset.merge(total_stops, on="Train_No")

# Remove any remaining NaN
final_dataset = final_dataset.dropna()

print("Training model...")

# Define features and target
X = final_dataset[["Distance", "Total_Stops"]]
y = final_dataset["Journey_Duration_Minutes"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as model.pkl")