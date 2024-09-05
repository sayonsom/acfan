import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import met_typical_tasks
from tqdm import tqdm

# Define the parameters for the distributions
air_temp_mean, air_temp_std = 22.5, 2.5  # in degrees Celsius (Normal Distribution)
air_velocity_shape, air_velocity_scale = 2, 0.2  # Gamma Distribution
humidity_alpha, humidity_beta = 2, 5  # Beta Distribution
clo_mean, clo_std = 1.0, 0.3  # in clo units (Normal Distribution)
heart_rate_mean, heart_rate_std = 75, 10  # in bpm (Normal Distribution)
skin_temp_mean, skin_temp_std = 33, 1  # in degrees Celsius (Normal Distribution)

# Number of data points to generate
num_samples = 500000

# Get current utc time in seconds
utc_time = int(pd.Timestamp.now().timestamp())

# Generate synthetic data
data = {
    'AirVelocity_AirTemperature': list(zip(
        np.random.gamma(air_velocity_shape, air_velocity_scale, num_samples),
        np.random.normal(air_temp_mean, air_temp_std, num_samples))),
    'Humidity': np.random.beta(humidity_alpha, humidity_beta, num_samples) * 100,  # Beta distribution scaled to 0-100
    'Clo': np.random.normal(clo_mean, clo_std, num_samples),
    'HeartRate': np.random.normal(heart_rate_mean, heart_rate_std, num_samples),
    'SkinTemperature': np.random.normal(skin_temp_mean, skin_temp_std, num_samples)
}

# Ensure values are within realistic bounds (clipping)
data['AirVelocity_AirTemperature'] = [(np.clip(av, 0, 1), np.clip(at, 15, 30)) for av, at in data['AirVelocity_AirTemperature']]
data['Humidity'] = np.clip(data['Humidity'], 20, 80)
data['Clo'] = np.clip(data['Clo'], 0.5, 1.5)
data['HeartRate'] = np.clip(data['HeartRate'], 50, 120)  # Assuming a range of 50-120 bpm for resting heart rate
data['SkinTemperature'] = np.clip(data['SkinTemperature'], 30, 36)  # Assuming a range of 30-36°C for skin temperature

# Function to estimate metabolic rate from heart rate
def estimate_met_from_hr(hr):
    # This is a simplified estimation and should be replaced with a more accurate model
    return (hr - 60) / 30 + 1  # Rough estimate: 1 MET at 60 bpm, increasing by 1 MET per 30 bpm increase

# Calculate PMV for each set of parameters
pmv_values = []
for i in tqdm(range(num_samples)):
    av, at = data['AirVelocity_AirTemperature'][i]
    h = data['Humidity'][i]
    clo = data['Clo'][i]
    hr = data['HeartRate'][i]
    st = data['SkinTemperature'][i]
    
    # Estimate metabolic rate from heart rate
    met = estimate_met_from_hr(hr)
    
    # Use the updated PMV calculation with both tdb and tr
    pmv_result = pmv_ppd(tdb=at, tr=at, vr=av, rh=h, met=met, clo=clo, wme=0)
    pmv_values.append(pmv_result['pmv'])

data['PMV'] = pmv_values

# Filter the dataframe such that only PMV values between -0.01 and 0.01 are included
data = {key: [value for i, value in enumerate(data[key]) if -0.01 <= data['PMV'][i] <= 0.01] for key in data}

# Convert to DataFrame
df = pd.DataFrame(data)
df = df.dropna()

# Save to CSV
df.to_csv(f'synthetic_pmv_data_{num_samples}_{utc_time}.csv', index=False)
df.to_csv('synthetic_pmv_data.csv', index=False)

print(f"Synthetic data generated and saved to 'synthetic_pmv_data_{num_samples}_{utc_time}.csv'")