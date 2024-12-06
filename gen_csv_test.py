import csv
import time
import random
from datetime import datetime

# Define the file path for the test CSV
test_csv_path = "data/test_log.csv"

# Generate or append data to the test CSV
def append_to_csv(file_path):
    # Check if the file exists
    try:
        with open(file_path, 'r') as f:
            pass
    except FileNotFoundError:
        # If the file doesn't exist, create it and write the headers
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'control_module_voltage', 'engine_rpm', 'engine_coolant_temp'])

    # Open the file in append mode and add new data
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Generate random data
        current_time = datetime.now().timestamp()
        engine_speed = 0
        control_module_voltage = round(random.uniform(11.0, 14.0), 2)  # Random voltage between 11V and 14V
        engine_rpm = random.randint(700, 3000)  # Random RPM between 700 and 3000
        engine_coolant_temp = random.randint(70, 120)  # Random temperature between 70°C and 120°C
        
        # Write the new row to the CSV
        writer.writerow([current_time, engine_speed, control_module_voltage, engine_rpm, engine_coolant_temp])
        print(f"Appended data: {current_time}, {engine_speed}, {control_module_voltage}, {engine_rpm}, {engine_coolant_temp}")

# Run the program to append data every second
if __name__ == "__main__":
    print("Generating test CSV data...")
    try:
        while True:
            append_to_csv(test_csv_path)
            time.sleep(1)  # Append new data every second
    except KeyboardInterrupt:
        print("Stopped data generation.")
