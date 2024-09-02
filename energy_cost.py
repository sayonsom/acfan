import numpy as np
from scipy.optimize import minimize

# Example cost function to minimize energy while maintaining comfort
def energy_cost(params, T_skin, HR, RH, clo, T_outdoor, RH_outdoor, activity_stage):
    v_fan, v_ac, T_ac_setpoint = params
    
    # Calculate total air velocity
    v_total = v_fan + v_ac
    
    # Comfort temperature calculation (simplified for demonstration)
    T_comfort = (0.5 * T_skin) + (-0.02 * HR) + (1.5 / v_total) + (-0.1 * np.log(RH)) + (0.8 * clo) + 18
    
    # Energy consumption (simplified model)
    P_ac = (T_ac_setpoint - T_outdoor) * v_ac  # Placeholder for actual AC power function
    P_fan = v_fan * 10  # Placeholder for actual fan power function (lower multiplier)
    
    # Penalize deviations from comfort temperature
    comfort_penalty = np.abs(T_comfort - T_ac_setpoint)
    
    # Total cost is a weighted sum of energy and discomfort
    cost = P_ac + P_fan + 100 * comfort_penalty  # Adjust weights as needed
    
    return cost

# Initial parameters for optimization
initial_params = [1.0, 1.0, 24.0]  # v_fan, v_ac, T_ac_setpoint

# Optimization call
result = minimize(energy_cost, initial_params, args=(33.5, 60, 50, 0.5, 30, 70, 'sleeping'))

# Optimal parameters
optimal_v_fan, optimal_v_ac, optimal_T_ac_setpoint = result.x


# print all the input parameters and the optimal values
print("Input Parameters:")
print("Skin Temperature: 33.5 °C")
print("Heart Rate: 60 bpm")
print("Relative Humidity: 50%")
print("Clothing Insulation: 0.5")
print("Outdoor Temperature: 30 °C")

print(f"Optimal Fan Velocity: {optimal_v_fan} m/s")
print(f"Optimal AC Air Velocity: {optimal_v_ac} m/s")
print(f"Optimal AC Setpoint Temperature: {optimal_T_ac_setpoint} °C")


