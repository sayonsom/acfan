import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt

# Parameters
num_nights = 1  # We will generate data for one night to plot
sleep_duration_minutes = 6 * 60  # 6 hours
sampling_rate_minutes = 1  # 1 minute resolution

# Temperature parameters
min_temp = 33.3
max_temp = 36.0

# Heart rate parameters
avg_heart_rate = 73  # bpm, based on provided data
heart_rate_variation = 3  # bpm for normal sleep
wake_heart_rate = 120  # bpm, significant increase when waking up
minutes_before_wake = 10  # Heart rate starts increasing 10 minutes before wake up

# Function to generate complex temperature data pattern for one night
def generate_complex_temp_pattern(start_time):
    time_points = [start_time + timedelta(minutes=i) for i in range(sleep_duration_minutes)]
    
    # Creating a pattern similar to the image
    segment_length = sleep_duration_minutes // 4
    temp_pattern = np.concatenate([
        np.linspace(34.0, 35.5, segment_length) + np.random.normal(0, 0.1, segment_length),
        np.linspace(35.5, 34.8, segment_length) + np.random.normal(0, 0.1, segment_length),
        np.linspace(34.8, 35.3, segment_length) + np.random.normal(0, 0.1, segment_length),
        np.linspace(35.3, 34.5, segment_length) + np.random.normal(0, 0.1, segment_length),
    ])
    
    return pd.DataFrame({
        'Timestamp': time_points,
        'Skin Temperature (°C)': temp_pattern
    })

# Function to generate synthetic heart rate data for one night with waking pattern
def generate_heart_rate_data(start_time):
    time_points = [start_time + timedelta(minutes=i) for i in range(sleep_duration_minutes)]
    heart_rates = np.random.normal(avg_heart_rate, heart_rate_variation, sleep_duration_minutes)
    
    # Create a waking up pattern
    for i in range(sleep_duration_minutes):
        if i < sleep_duration_minutes - minutes_before_wake:  # Stable heart rate until a few minutes before wake up
            if np.random.rand() < 0.1:  # Occasional small spikes
                heart_rates[i] += random.uniform(5, 10)
        else:  # Increase heart rate rapidly before waking up
            heart_rates[i] = np.linspace(avg_heart_rate, wake_heart_rate, minutes_before_wake)[i - (sleep_duration_minutes - minutes_before_wake)]
    
    return pd.DataFrame({
        'Timestamp': time_points,
        'Heart Rate (bpm)': heart_rates
    })

# Generate data for one night
start_hour = random.randint(22, 23)  # Start between 10 PM (22:00) and 11 PM (23:00)
start_minute = random.randint(0, 59)
start_time = datetime.combine(datetime.today(), datetime.min.time()) + timedelta(hours=start_hour, minutes=start_minute)

night_temperature_data = generate_complex_temp_pattern(start_time)
night_heart_rate_data = generate_heart_rate_data(start_time)

# Combine data into one DataFrame
combined_data = pd.merge(night_temperature_data, night_heart_rate_data, on='Timestamp')

# Save the combined data to a CSV file
output_file = 'sleep_skin_temp_heart_rate_data.csv'
combined_data.to_csv(output_file, index=False)

# Plot the data
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(combined_data['Timestamp'], combined_data['Skin Temperature (°C)'], label='Skin Temperature')
plt.xlabel('Time')
plt.ylabel('Skin Temperature (°C)')
plt.title('Synthetic Sleep Skin Temperature Data for One Night')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(combined_data['Timestamp'], combined_data['Heart Rate (bpm)'], label='Heart Rate', color='orange')
plt.xlabel('Time')
plt.ylabel('Heart Rate (bpm)')
plt.title('Synthetic Sleep Heart Rate Data for One Night')
plt.grid(True)

plt.tight_layout()
plt.show()
