import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from pythermalcomfort.models import pmv_ppd
import warnings
warnings.filterwarnings("ignore")


# Load the combined model and scaler
def load_model_and_scaler(data_size, synthetic_data_version):
    model = tf.keras.models.load_model(f'models/combined_model_data_{data_size}.keras')
    scaler = joblib.load(f'scalers/scaler_data_{data_size}.save')
    return model, scaler


def find_air_temperature_with_constraint(pmv_target, clo, humidity, predicted_temp, predicted_velocity, target_velocity=0.1, tolerance=0.01, max_iterations=100):
    min_temp, max_temp = predicted_temp - 10, predicted_temp + 10
    
    for _ in range(max_iterations):
        mid_temp = (min_temp + max_temp) / 2
        calculated_pmv = pmv_ppd(tdb=mid_temp, tr=mid_temp, vr=target_velocity, 
                                 rh=humidity, met=1.2, clo=clo)['pmv']
        
        if abs(calculated_pmv - pmv_target) < tolerance:
            return mid_temp
        elif calculated_pmv < pmv_target:
            min_temp = mid_temp
        else:
            max_temp = mid_temp
    
    return mid_temp

def predict_temp_and_velocity(model, scaler, pmv, heart_rate, skin_temp, clo, humidity):
    input_data = np.array([[pmv, heart_rate, clo, humidity, skin_temp]])
    scaled_input = scaler.transform(input_data)
    predicted_velocity, predicted_temp = model.predict(scaled_input)[0]
    return predicted_temp, predicted_velocity

def calculate_energy_consumption(initial_temp, target_temp, outside_temp, room_size, insulation, air_velocity):
    # Constants (you may need to adjust these based on your specific scenario)
    air_density = 1.225  # kg/m^3
    specific_heat_capacity = 1005  # J/(kg*K)
    cop = 3.5  # Coefficient of Performance for the AC system
    
    # Calculate temperature difference
    delta_T = abs(initial_temp - target_temp)
    
    # Calculate heat transfer through walls (simplified)
    wall_area = (room_size[0] * room_size[1] * 2) + (room_size[1] * room_size[2] * 2) + (room_size[0] * room_size[2] * 2)
    heat_transfer = wall_area * insulation * (outside_temp - target_temp)
    
    # Calculate energy needed to change air temperature
    room_volume = room_size[0] * room_size[1] * room_size[2]
    air_mass = air_density * room_volume
    energy_for_temp_change = air_mass * specific_heat_capacity * delta_T
    
    # Calculate energy for air movement
    fan_power = 50  # Assuming a 50W fan, adjust as needed
    fan_energy = fan_power * air_velocity  # Simplified relationship
    
    # Total energy
    total_energy = (energy_for_temp_change + heat_transfer) / cop + fan_energy
    
    return total_energy

def find_optimal_setpoints(model, scaler, initial_temp, outside_temp, humidity, room_size, insulation, pmv_target=0):
    clo = 0.5  # Assuming a standard clothing insulation, adjust if needed
    heart_rate = 75  # Assuming an average heart rate, adjust if needed
    skin_temp = 33  # Assuming an average skin temperature, adjust if needed
    
    # Predict temperature and velocity without constraint
    predicted_temp, predicted_velocity = predict_temp_and_velocity(
        model, scaler, pmv_target, heart_rate, skin_temp, clo, humidity
    )
    
    # Find temperature with air velocity constraint
    target_velocity = 0.1
    predicted_temp_constrained = find_air_temperature_with_constraint(
        pmv_target, clo, humidity, predicted_temp, predicted_velocity, target_velocity=target_velocity
    )
    
    # Calculate energy consumption for both scenarios
    energy_unconstrained = calculate_energy_consumption(
        initial_temp, predicted_temp, outside_temp, room_size, insulation, predicted_velocity
    )
    energy_constrained = calculate_energy_consumption(
        initial_temp, predicted_temp_constrained, outside_temp, room_size, insulation, target_velocity
    )
    
    # Choose the setpoints with lower energy consumption
    if energy_unconstrained < energy_constrained:
        return predicted_temp, predicted_velocity, energy_unconstrained
    else:
        return predicted_temp_constrained, target_velocity, energy_constrained

if __name__ == "__main__":
    # Load the model and scaler
    data_size = "500000"
    synthetic_data_version = "1725514933"
    model, scaler = load_model_and_scaler(data_size, synthetic_data_version)
    
    # Get user inputs
    initial_temp = float(input("Enter initial room temperature (°C): "))
    outside_temp = float(input("Enter outside temperature (°C): "))
    humidity = float(input("Enter relative humidity (%): "))
    length = float(input("Enter room length (m): "))
    width = float(input("Enter room width (m): "))
    height = float(input("Enter room height (m): "))
    insulation = float(input("Enter room insulation U-value (W/m²K): "))
    
    room_size = (length, width, height)
    
    # Find optimal setpoints
    optimal_temp, optimal_velocity, energy_consumption = find_optimal_setpoints(
        model, scaler, initial_temp, outside_temp, humidity, room_size, insulation
    )
    
    # Display results
    print("\nOptimal Setpoints:")
    print(f"AC Temperature Setpoint: {optimal_temp:.2f}°C")
    print(f"Air Velocity Setpoint: {optimal_velocity:.2f} m/s")
    print(f"Estimated Energy Consumption: {energy_consumption:.2f} J")