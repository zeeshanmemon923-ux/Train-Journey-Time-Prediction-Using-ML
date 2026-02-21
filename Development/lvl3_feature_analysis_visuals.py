import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset1.csv")

df = df.drop_duplicates()

# Fix time columns
df["Arrival_time"] = df["Arrival_time"].replace(["Source"], pd.NA)
df["Departure_Time"] = df["Departure_Time"].replace(["Destination"], pd.NA)

df["Arrival_time"] = pd.to_datetime(df["Arrival_time"], errors="coerce")
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], errors="coerce")

df["Arrival_Minutes"] = df["Arrival_time"].dt.hour * 60 + df["Arrival_time"].dt.minute
df["Departure_Minutes"] = df["Departure_Time"].dt.hour * 60 + df["Departure_Time"].dt.minute

df = df.sort_values(["Train_No", "Distance"])

# Calculate duration
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

# Feature creation
total_distance = df.groupby("Train_No")["Distance"].max().reset_index()
total_stops = df.groupby("Train_No").size().reset_index(name="Total_Stops")

final_dataset = journey_times.merge(total_distance, on="Train_No")
final_dataset = final_dataset.merge(total_stops, on="Train_No")

# TASK 3.1 - Distance vs Duration

plt.figure()
plt.scatter(final_dataset["Distance"], final_dataset["Journey_Duration_Minutes"])
plt.xlabel("Total Distance")
plt.ylabel("Journey Duration (Minutes)")
plt.title("Distance vs Journey Duration")
plt.show()

# TASK 3.2 - Stops vs Duration

plt.figure()
plt.scatter(final_dataset["Total_Stops"], final_dataset["Journey_Duration_Minutes"])
plt.xlabel("Total Stops")
plt.ylabel("Journey Duration (Minutes)")
plt.title("Stops vs Journey Duration")
plt.show()

# TASK 3.3 - Correlation Matrix

correlation = final_dataset[["Distance", "Total_Stops", "Journey_Duration_Minutes"]].corr()
print("\n===== CORRELATION MATRIX =====")
print(correlation)

# TASK 3.4 - Pivot Table

pivot_table = df.pivot_table(
    index="Train_No",
    values="Distance",
    aggfunc="count"
)

print("\n===== PIVOT TABLE (Stops per Train) =====")
print(pivot_table.head())
