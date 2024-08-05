import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from utilities.conversions import fan_speed_to_air_velocity

def calculate_surface_temperature(T_i, T_o, R_i, R_o):
    return (T_i * R_i + T_o * R_o) / (R_i + R_o)

def calculate_MRT(surface_temps, surface_areas, direction='north', curtains=False):
    total_area = sum(surface_areas)
    weighted_temp = sum(T * A for T, A in zip(surface_temps, surface_areas))
    if direction.lower() == 'south':
        solar_gain_factor = 1.05
    elif direction.lower() == 'west':
        solar_gain_factor = 1.04
    elif direction.lower() == 'east':
        solar_gain_factor = 1.03
    else:  # north
        solar_gain_factor = 1.01

    mrt = (solar_gain_factor * weighted_temp) / total_area
    return mrt

def estimate_window_area(num_windows, window_height, window_width):
    return num_windows * window_height * window_width

def convert_ft_to_m(feet):
    return feet * 0.3048

def get_user_input(outdoor_temp, humidity, direction):
    ac_cop = 3  # coefficient of performance of the AC
    fan_power = 0.075  # power of the ceiling fan in kW (75 watts)
    indoor_temp = outdoor_temp - 4.5
    initial_temp = indoor_temp  # initial room temperature in °C
    pmv = 0
    clo = 1.0
    ac_power = 3.5  # 1 ton of AC in kW
    fan_speed_setting = 3
    floor_length_ft = 10
    floor_width_ft = 10
    ceiling_height_ft = 10
    num_windows = 2
    window_height_ft = 3
    window_width_ft = 3
    curtains = True
    air_velocity = fan_speed_to_air_velocity(fan_speed_setting, ceiling_height=ceiling_height_ft, room_area=floor_length_ft * floor_width_ft)
    
    floor_length = convert_ft_to_m(floor_length_ft)
    floor_width = convert_ft_to_m(floor_width_ft)
    ceiling_height = convert_ft_to_m(ceiling_height_ft)
    window_height = convert_ft_to_m(window_height_ft)
    window_width = convert_ft_to_m(window_width_ft)
    
    room_area = floor_length * floor_width
    window_area = estimate_window_area(num_windows, window_height, window_width)
    
    wall_thermal_resistance = 0.5
    window_thermal_resistance = 0.8 if not curtains else 1.5
    
    wall_area = (2 * ceiling_height * (floor_length + floor_width)) - window_area
    ceiling_area = floor_length * floor_width
    floor_area = ceiling_area

    wall_temp = calculate_surface_temperature(indoor_temp, outdoor_temp, wall_thermal_resistance, wall_thermal_resistance)
    window_temp = calculate_surface_temperature(indoor_temp, outdoor_temp, window_thermal_resistance, window_thermal_resistance)
    ceiling_temp = indoor_temp
    floor_temp = indoor_temp

    surface_temps = [wall_temp, window_temp, ceiling_temp, floor_temp]
    surface_areas = [wall_area, window_area, ceiling_area, floor_area]

    room_volume = floor_length * floor_width * ceiling_height

    mrt = calculate_MRT(surface_temps, surface_areas, direction, curtains)
    # AirTemperature,RadiantTemperature,AirVelocity,Humidity,Clo,PMV
    data = {
        'RadiantTemperature': [mrt],
        'AirVelocity': [air_velocity],
        'Humidity': [humidity],
        'Clo': [clo],
        'PMV': [pmv]
    }
    df = pd.DataFrame(data)
    
    return df, initial_temp, room_volume, ac_power, ac_cop, fan_power

def predict_cooling_time_and_energy(ac_power, ac_cop, fan_power, initial_temp, target_temp, room_volume):
    delta_temp = initial_temp - target_temp
    cooling_load_kwh = (room_volume * delta_temp * 0.02)  # 0.02 is a rough estimate for specific heat capacity and air density
    ac_energy = cooling_load_kwh / ac_cop
    ac_time_hours = cooling_load_kwh / ac_power

    # AC + Fan scenario
    adjusted_temp_delta = delta_temp / 1.2  # Fan effect, assuming 20% increase in perceived cooling
    cooling_load_kwh_fan = (room_volume * adjusted_temp_delta * 0.02)
    ac_energy_fan = cooling_load_kwh_fan / ac_cop
    ac_time_fan_hours = cooling_load_kwh_fan / ac_power
    fan_energy = fan_power * ac_time_fan_hours

    return ac_energy, ac_time_hours, ac_energy_fan + fan_energy, ac_time_fan_hours

data_size = 500000
synthetic_data_version = 1722757024

model_name = f'models/synthetic_pmv_data_{data_size}_{synthetic_data_version}.h5'
model = tf.keras.models.load_model(model_name)
scaler = joblib.load(f'scalers/scaler_{data_size}_{synthetic_data_version}.save')

scenarios = [25, 30, 35, 40, 45]
directions = ['north', 'south', 'east', 'west']
humidity_range = np.arange(10, 61, 1)

results = []

for outdoor_temp in scenarios:
    for humidity in humidity_range:
        for direction in directions:
            input_data, initial_temp, room_volume, ac_power, ac_cop, fan_power = get_user_input(outdoor_temp, humidity, direction)
            input_features = input_data[['RadiantTemperature', 'AirVelocity', 'Humidity', 'Clo', 'PMV']]
            input_scaled = scaler.transform(input_features)
            predicted_air_temperature = model.predict(input_scaled)
            target_temp = predicted_air_temperature[0][0]

            ac_energy, ac_time, ac_fan_energy, ac_fan_time = predict_cooling_time_and_energy(ac_power, ac_cop, fan_power, initial_temp, target_temp, room_volume)

            results.append([outdoor_temp, humidity, direction, ac_energy, ac_time, ac_fan_energy, ac_fan_time, target_temp])

# Create DataFrame from results
results_df = pd.DataFrame(results, columns=['OutdoorTemperature', 'Humidity', 'Direction', 'AC_Energy_kWh', 'AC_Time_hours', 'AC_Fan_Energy_kWh', 'AC_Fan_Time_hours', 'PredictedTemperature'])

# Save to CSV
results_df.to_csv('cooling_scenarios_results.csv', index=False)

# Plot results
for outdoor_temp in scenarios:
    for direction in directions:
        subset = results_df[(results_df['OutdoorTemperature'] == outdoor_temp) & (results_df['Direction'] == direction)]
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12))

        # Energy Consumption Plot
        ax1.set_xlabel('Humidity (%)')
        ax1.set_ylabel('Energy Consumption (kWh)', color='tab:blue')
        ax1.plot(subset['Humidity'], subset['AC_Energy_kWh'], label='AC Only Energy', color='blue')
        ax1.plot(subset['Humidity'], subset['AC_Fan_Energy_kWh'], label='AC + Fan Energy', color='red')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Secondary axis (right y-axis) for predicted temperature
        ax2 = ax1.twinx()
        ax2.set_ylabel('Predicted Temperature (°C)', color='tab:green')
        ax2.plot(subset['Humidity'], subset['PredictedTemperature'], label='Predicted Temperature', color='green', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout(pad=3.0)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title(f'Energy Consumption and Predicted Temperature vs Humidity at {outdoor_temp}°C ({direction.capitalize()} Facing)')

        # Cooling Time Plot
        ax3.set_xlabel('Humidity (%)')
        ax3.set_ylabel('Cooling Time (hours)', color='tab:blue')
        ax3.plot(subset['Humidity'], subset['AC_Time_hours'], label='AC Only Time', color='blue')
        ax3.plot(subset['Humidity'], subset['AC_Fan_Time_hours'], label='AC + Fan Time', color='red')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

        # Secondary axis (right y-axis) for predicted temperature
        ax4 = ax3.twinx()
        ax4.set_ylabel('Predicted Temperature (°C)', color='tab:green')
        ax4.plot(subset['Humidity'], subset['PredictedTemperature'], label='Predicted Temperature', color='green', linestyle='--')
        ax4.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout(pad=3.0)
        ax3.legend(loc='upper left')
        ax4.legend(loc='upper right')
        ax3.set_title(f'Cooling Time and Predicted Temperature vs Humidity at {outdoor_temp}°C ({direction.capitalize()} Facing)')

        plt.savefig(f'cooling_scenario_{outdoor_temp}_{direction}.png')
        plt.show()

print('CSV results generated and plots saved.')
