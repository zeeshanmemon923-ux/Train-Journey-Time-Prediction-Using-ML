
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Dataset1.csv")
df = df.drop_duplicates()

df["Arrival_time"] = df["Arrival_time"].replace(["Source"], pd.NA)
df["Departure_Time"] = df["Departure_Time"].replace(["Destination"], pd.NA)

df["Arrival_time"] = pd.to_datetime(df["Arrival_time"], errors="coerce")
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], errors="coerce")


df["Arrival_Minutes"] = df["Arrival_time"].dt.hour * 60 + df["Arrival_time"].dt.minute
df["Departure_Minutes"] = df["Departure_Time"].dt.hour * 60 + df["Departure_Time"].dt.minute


df = df.sort_values(["Train_No", "Distance"])

journey_times = df.groupby("Train_No").agg(
    Start_Time=("Departure_Minutes", "first"),
    End_Time=("Arrival_Minutes", "last")
).reset_index()

journey_times["Journey_Duration_Minutes"] = (
    journey_times["End_Time"] - journey_times["Start_Time"]
)

journey_times.loc[
    journey_times["Journey_Duration_Minutes"] < 0,
    "Journey_Duration_Minutes"
] += 24 * 60

total_distance = df.groupby("Train_No")["Distance"].max().reset_index()
total_stops = df.groupby("Train_No").size().reset_index(name="Total_Stops")

final_dataset = journey_times.merge(total_distance, on="Train_No")
final_dataset = final_dataset.merge(total_stops, on="Train_No")

final_dataset = final_dataset.dropna(subset=["Journey_Duration_Minutes"])

# Train Final Model (Improved)

X = final_dataset[["Distance", "Total_Stops"]]
y = final_dataset["Journey_Duration_Minutes"]

model = LinearRegression()
model.fit(X, y)

print("\nModel trained successfully.")

# Interactive Prediction System

while True:
    print("\n--- Train Journey Duration Predictor ---")
    
    try:
        distance = float(input("Enter total journey distance (km): "))
        stops = int(input("Enter total number of stops: "))
        
        prediction = model.predict([[distance, stops]])
        predicted_minutes = prediction[0]

        hours = int(predicted_minutes // 60)
        minutes = int(predicted_minutes % 60)

        print(f"\nEstimated Journey Duration: {hours} hours {minutes} minutes")

    except:
        print("Invalid input. Please enter numeric values.")

    again = input("\nDo you want to predict again? (yes/no): ")
    if again.lower() != "yes":
        break

print("\nProgram ended.")
