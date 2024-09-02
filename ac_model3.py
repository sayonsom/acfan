import pandas as pd

# Assuming the previously defined functions are available in the same environment:
# - calculate_pmv
# - optimize_feels_like_temperature

# Load the dataset
file_path = '/mnt/data/file-RYm7ctBreEMbKkGacL5KPo9p'
df = pd.read_csv(file_path)

# Initialize some default values for parameters
clothing_insulation = 0.5  # clo, typical for light clothing
metabolic_rate = 1.2  # met, example for light work
max_air_velocity = 3.0  # m/s, maximum air velocity

# Initialize lists to store the results
adjusted_temps = []
adjusted_humidities = []
adjusted_air_velocities = []
pmv_values = []

# Loop through the dataframe
for i in range(len(df) - 1):
    # Current conditions
    current_temp = df.loc[i, 'SkinTemperature']
    current_humidity = df.loc[i, 'Humidity(%)']
    actual_temp = df.loc[i, 'Outside_Temperature(Celcius)']
    
    # Forecasted conditions (next row)
    forecast_temp = df.loc[i + 1, 'Outside_Temperature(Celcius)']
    forecast_humidity = df.loc[i + 1, 'Humidity(%)']
    
    # Desired PMV (close to 0) and optimized feels-like temperature (same as forecasted)
    optimized_temp = actual_temp  # Here we're using the actual temp as a reference for the optimization

    # Optimize the air velocity to achieve the desired feels-like temperature
    optimal_air_velocity, _ = optimize_feels_like_temperature(
        actual_temp=actual_temp,
        humidity=current_humidity,
        metabolic_rate=metabolic_rate,
        skin_temp=current_temp,
        optimized_temp=optimized_temp,
        max_air_velocity=max_air_velocity
    )
    
    # Calculate the PMV with the optimized settings
    pmv_value = calculate_pmv(
        ta=actual_temp,
        tr=actual_temp,  # Assuming mean radiant temp is same as air temp
        rh=current_humidity,
        met=metabolic_rate,
        clo=clothing_insulation,
        air_velocity=optimal_air_velocity
    )
    
    # Adjust setpoint based on current and forecast data
    adjusted_temps.append(optimized_temp)
    adjusted_humidities.append(current_humidity)
    adjusted_air_velocities.append(optimal_air_velocity)
    pmv_values.append(pmv_value)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Time': df['HHMMSS'].iloc[:-1],
    'Adjusted_Temperature': adjusted_temps,
    'Adjusted_Humidity': adjusted_humidities,
    'Adjusted_Air_Velocity': adjusted_air_velocities,
    'PMV': pmv_values
})

# Save results to a new CSV file
results_file_path = '/mnt/data/adjusted_pmv_results.csv'
results_df.to_csv(results_file_path, index=False)

print(f"Results saved to {results_file_path}")
