import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# TASK 5.1 - Basic Model (Distance Only)

X_basic = final_dataset[["Distance"]]
y = final_dataset["Journey_Duration_Minutes"]

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

basic_model = LinearRegression()
basic_model.fit(X_train_b, y_train)

y_pred_basic = basic_model.predict(X_test_b)

mae_basic = mean_absolute_error(y_test, y_pred_basic)
rmse_basic = np.sqrt(mean_squared_error(y_test, y_pred_basic))

print("\n===== TASK 5.1 - BASIC MODEL RESULTS =====")
print("MAE:", mae_basic)
print("RMSE:", rmse_basic)

# TASK 5.2 - Improved Model (Distance + Stops)

X_improved = final_dataset[["Distance", "Total_Stops"]]

X_train_i, X_test_i, y_train, y_test = train_test_split(
    X_improved, y, test_size=0.2, random_state=42
)

improved_model = LinearRegression()
improved_model.fit(X_train_i, y_train)

y_pred_improved = improved_model.predict(X_test_i)

mae_improved = mean_absolute_error(y_test, y_pred_improved)
rmse_improved = np.sqrt(mean_squared_error(y_test, y_pred_improved))

print("\n===== TASK 5.2 - IMPROVED MODEL RESULTS =====")
print("MAE:", mae_improved)
print("RMSE:", rmse_improved)


# TASK 5.3 - Model Comparison

print("\n===== TASK 5.3 - COMPARISON =====")
print("Basic Model RMSE:", rmse_basic)
print("Improved Model RMSE:", rmse_improved)

# TASK 5.4 - Select Best Model

if rmse_improved < rmse_basic:
    print("\nImproved Model performs better.")
else:
    print("\nBasic Model performs better.")
