import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
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

def get_clo_value():
    print("Select clothing insulation level:")
    print("1. Light indoor clothing (0.5 clo)")
    print("2. Typical indoor clothing (1.0 clo)")
    print("3. Light pajamas (0.5 clo)")
    print("4. Warm pajamas (0.7 clo)")
    print("5. Bedsheet (0.2 clo)")
    print("6. Light blanket (0.5 clo)")
    print("7. Heavy blanket or quilt (1.0 clo)")
    print("8. Down comforter (1.5 clo)")
    print("9. Pajamas + Light blanket (1.2 clo)")
    print("10. Pajamas + Heavy blanket (1.5 clo)")
    print("11. Pajamas + Down comforter (2.0 clo)")
    choice = int(input("Enter the number corresponding to your choice: "))
    clo_values = [0.5, 1.0, 0.5, 0.7, 0.2, 0.5, 1.0, 1.5, 1.2, 1.5, 2.0]
    return clo_values[choice - 1]

def get_user_input():
    ac_cop = 3  # coefficient of performance of the AC
    fan_power = 0.075  # power of the ceiling fan in kW (75 watts)
    outdoor_temp = float(input("Enter Outdoor Temperature (°C): "))
    indoor_temp = outdoor_temp - 4.5
    initial_temp = indoor_temp  # initial room temperature in °C
    pmv = 0
    humidity = float(input("Enter Humidity (%): "))
    clo = get_clo_value()
    ac_power = float(input("Enter Tons of AC (1 ton = 3.5 kW): ")) * 3.5
    fan_speed_setting = int(input("Enter Fan Speed Setting (0-5): "))
    floor_length_ft = float(input("Enter Floor Length (ft): "))
    floor_width_ft = float(input("Enter Floor Width (ft): "))
    ceiling_height_ft = float(input("Enter Ceiling Height (ft): "))
    num_windows = int(input("Enter Number of Windows: "))
    window_height_ft = float(input("Enter Window Height (ft): "))
    window_width_ft = float(input("Enter Window Width (ft): "))
    direction = input("Is the window facing North, South, East, or West? ").strip().lower()
    curtains = input("Are curtains on? (yes/no): ").strip().lower() == "yes"
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

input_data, initial_temp, room_volume, ac_power, ac_cop, fan_power = get_user_input()

# Only include features used during training
input_features = input_data[['RadiantTemperature', 'AirVelocity', 'Humidity', 'Clo', 'PMV']]
input_scaled = scaler.transform(input_features)
predicted_air_temperature = model.predict(input_scaled)

target_temp = predicted_air_temperature[0][0]

print(f'Input Data:')
print(input_data)

print(f'Predicted Air Temperature: {target_temp:.2f}°C')

ac_energy, ac_time, ac_fan_energy, ac_fan_time = predict_cooling_time_and_energy(ac_power, ac_cop, fan_power, initial_temp, target_temp, room_volume)

print(f'AC Only: Energy Consumption: {ac_energy:.2f} kWh, Cooling Time: {ac_time:.2f} hours')
print(f'AC + Fan: Energy Consumption: {ac_fan_energy:.2f} kWh, Cooling Time: {ac_fan_time:.2f} hours')
