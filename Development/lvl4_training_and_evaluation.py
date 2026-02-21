
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

# TASK 4.1 - Split Dataset

X = final_dataset[["Distance", "Total_Stops"]]
y = final_dataset["Journey_Duration_Minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TASK 4.2 - Train Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)

# TASK 4.3 - Evaluate Model

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("RMSE:", rmse)

# TASK 4.4 - Visualize Predictions

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Duration")
plt.ylabel("Predicted Duration")
plt.title("Actual vs Predicted Journey Duration")
plt.show()
