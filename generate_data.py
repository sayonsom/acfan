import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from tqdm import tqdm

# Define the parameters for the distributions
air_temp_mean, air_temp_std = 22.5, 2.5  # in degrees Celsius (Normal Distribution)
radiant_temp_mean, radiant_temp_std = 22.5, 2.5  # in degrees Celsius (Normal Distribution)
air_velocity_shape, air_velocity_scale = 2, 0.2  # Gamma Distribution
humidity_alpha, humidity_beta = 2, 5  # Beta Distribution
clo_mean, clo_std = 1.0, 0.3  # in clo units (Normal Distribution)

# Number of data points to generate
num_samples = 500000

# Get current utc time in seconds
utc_time = int(pd.Timestamp.now().timestamp())

# Generate synthetic data
data = {
    'AirTemperature': np.random.normal(air_temp_mean, air_temp_std, num_samples),
    'RadiantTemperature': np.random.normal(radiant_temp_mean, radiant_temp_std, num_samples),
    'AirVelocity': np.random.gamma(air_velocity_shape, air_velocity_scale, num_samples),
    'Humidity': np.random.beta(humidity_alpha, humidity_beta, num_samples) * 100,  # Beta distribution scaled to 0-100
    'Clo': np.random.normal(clo_mean, clo_std, num_samples)
}

# Ensure values are within realistic bounds (clipping)
data['AirTemperature'] = np.clip(data['AirTemperature'], 15, 30)
data['RadiantTemperature'] = np.clip(data['RadiantTemperature'], 15, 30)
data['AirVelocity'] = np.clip(data['AirVelocity'], 0, 1)
data['Humidity'] = np.clip(data['Humidity'], 20, 80)
data['Clo'] = np.clip(data['Clo'], 0.5, 1.5)

# Calculate PMV for each set of parameters
pmv_values = []
for i in tqdm(range(num_samples)):
    at = data['AirTemperature'][i]
    rt = data['RadiantTemperature'][i]
    av = data['AirVelocity'][i]
    h = data['Humidity'][i]
    clo = data['Clo'][i]
    
    # Use a standard metabolic rate and activity level (1.2 met and 0.0 external work)
    pmv_result = pmv_ppd(tdb=at, tr=rt, vr=av, rh=h, met=1.2, clo=clo, wme=0.0)
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
