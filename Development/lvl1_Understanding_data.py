import pandas as pd

# Task 1.1 Total records and columns

# Load dataset
df = pd.read_csv("Dataset1.csv")

# Check total records and columns           
print("Dataset Shape:", df.shape)

print("Total Records (Rows):", df.shape[0])
print("Total Columns:", df.shape[1])

# Display column names
print("\nColumn Names:")
print(df.columns)

# TASK 1.2 - Train-wise starting and ending stations

# Sort dataset by Train_No and Distance
df = df.sort_values(["Train_No", "Distance"])

# Get starting station for each train
starting_station = df.groupby("Train_No").first()["Station_Name"]

# Get ending station for each train
ending_station = df.groupby("Train_No").last()["Station_Name"]

# Combine into one table
train_table = pd.DataFrame({
    "Starting_Station": starting_station,
    "Ending_Station": ending_station
})

print("\nTrain-wise Starting and Ending Stations:")
print(train_table.head())


# TASK 1.3 - Basic statistics for Distance

print("\n===== DISTANCE STATISTICS =====")
print(df["Distance"].describe())


# Calculate number of stops per train
stops_per_train = df.groupby("Train_No").size()

print("\n===== STOPS PER TRAIN STATISTICS =====")
print(stops_per_train.describe())

# TASK 1.4 - Data Quality Checks

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


print("\n===== DUPLICATE ROWS =====")
print("Total Duplicate Rows:", df.duplicated().sum())


print("\n===== CHECKING NEGATIVE DISTANCE VALUES =====")
print("Negative Distance Rows:", (df["Distance"] < 0).sum())        