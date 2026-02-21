
import pandas as pd

#TASK 2.1 Handle missing values and remove duplicate records

df = pd.read_csv("Dataset1.csv")

print("Columns in dataset:")
print(df.columns)

# Remove duplicates
df = df.drop_duplicates()

# Drop rows with missing values (if any)
df = df.dropna()

print("After cleaning:")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# TASK 2.2 - Convert Time to ML-Friendly Format

# Convert time columns to datetime
df["Arrival_time"] = df["Arrival_time"].replace(["Source"], pd.NA)
df["Departure_Time"] = df["Departure_Time"].replace(["Destination"], pd.NA)

# Convert to datetime (without strict format)
df["Arrival_time"] = pd.to_datetime(df["Arrival_time"], errors="coerce")
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], errors="coerce")

# Convert to total minutes
df["Arrival_Minutes"] = df["Arrival_time"].dt.hour * 60 + df["Arrival_time"].dt.minute
df["Departure_Minutes"] = df["Departure_Time"].dt.hour * 60 + df["Departure_Time"].dt.minute

print("\nTime successfully converted into numerical minutes.")
print(df[["Arrival_Minutes", "Departure_Minutes"]].head())

# TASK 2.3 - Calculate Journey Duration

# Sort properly
df = df.sort_values(["Train_No", "Distance"])

# Get first departure and last arrival per train
journey_times = df.groupby("Train_No").agg(
    Start_Time=("Departure_Minutes", "first"),
    End_Time=("Arrival_Minutes", "last")
).reset_index()

# Calculate duration
journey_times["Journey_Duration_Minutes"] = (
    journey_times["End_Time"] - journey_times["Start_Time"]
)

# Fix negative durations (overnight trains)
journey_times.loc[
    journey_times["Journey_Duration_Minutes"] < 0,
    "Journey_Duration_Minutes"
] += 24 * 60

print("\n===== JOURNEY DURATION (Minutes) =====")
print(journey_times.head())

# TASK 2.4 - Create Input Features

# Total distance per train
total_distance = df.groupby("Train_No")["Distance"].max().reset_index()

# Total stops per train
total_stops = df.groupby("Train_No").size().reset_index(name="Total_Stops")

# Merge with journey_times
final_dataset = journey_times.merge(total_distance, on="Train_No")
final_dataset = final_dataset.merge(total_stops, on="Train_No")

print("\n===== FINAL DATASET FOR MODELING =====")
print(final_dataset.head())
