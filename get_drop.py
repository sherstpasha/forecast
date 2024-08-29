import pandas as pd

# Step 1: Import the CSV data
# Replace 'weather_data.csv' with your actual file name
input_filename = "Meteo1-2023-5min.csv"
df = pd.read_csv(input_filename, delimiter=";")

# Step 2: Reduce the data to a 15-minute interval by keeping every third row
df_reduced = df.iloc[::3, :]

# Step 3: Convert the datetime column to the correct format
df_reduced["time_YYMMDD_HHMMSS"] = pd.to_datetime(
    df_reduced["time_YYMMDD_HHMMSS"], format="%y-%m-%d %H:%M:%S"
)

# Step 4: Save the reduced data to a new CSV file with a comma delimiter and the correct datetime format
output_filename = input_filename.replace(".csv", "_dropped.csv")
df_reduced.to_csv(output_filename, index=False, sep=",")

print(f"Reduced data saved to {output_filename}")
