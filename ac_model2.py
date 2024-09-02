import numpy as np
from math import exp

def calculate_pmv(ta, tr, rh, met=1.2, clo=0.5, air_velocity=0.1):
    """
    Calculate PMV (Predicted Mean Vote) based on the given parameters.
    - ta: Air temperature (°C)
    - tr: Mean radiant temperature (°C), assumed equal to ta if not provided
    - rh: Relative humidity (%)
    - met: Metabolic rate (met), default is 1.2 (sitting)
    - clo: Clothing insulation (clo), default is 0.5 (light clothing)
    - air_velocity: Air velocity (m/s), default is 0.1 m/s
    Returns PMV value.
    """
    pa = rh * 10 * exp(16.6536 - 4030.183 / (ta + 235))  # partial water vapor pressure

    icl = 0.155 * clo  # thermal insulation of clothing in m2K/W
    m = met * 58.15  # metabolic rate in W/m2
    w = 0  # external work in W/m2 (assumed to be zero)
    mw = m - w  # internal heat production in the human body

    fcl = 1.05 + 0.1 * icl if icl > 0.078 else 1 + 0.2 * icl
    hcf = 12.1 * np.sqrt(air_velocity)
    taa = ta + 273
    tra = tr + 273
    tcl = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * ((tra / 100) ** 4)
    xn = tcl / 100
    xf = tcl / 50
    n = 0
    eps = 0.00015
    
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        n += 1
        if n > 150:
            break

    tcl = 100 * xn - 273

    hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15)
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    return pmv

def optimize_conditions_to_pmv_zero(forecast_temp, forecast_humidity, current_temp, current_humidity, current_velocity):
    """
    Optimize the room temperature, humidity, and air velocity to achieve a PMV of 0.
    - forecast_temp: Forecasted outside temperature (°C)
    - forecast_humidity: Forecasted outside humidity (%)
    - current_temp: Current room temperature (°C)
    - current_humidity: Current room humidity (%)
    - current_velocity: Current air velocity in the room (m/s)
    Returns suggested room temperature, humidity, and air velocity to achieve PMV = 0.
    """
    tr = current_temp  # Assume mean radiant temperature is equal to air temperature
    target_pmv = 0
    max_iterations = 100
    tolerance = 0.01

    # Start with the current conditions
    room_temp = current_temp
    room_humidity = current_humidity
    air_velocity = current_velocity

    for _ in range(max_iterations):
        # Calculate current PMV
        current_pmv = calculate_pmv(room_temp, tr, room_humidity, air_velocity=air_velocity)

        # Check if PMV is within the tolerance
        if abs(current_pmv - target_pmv) <= tolerance:
            break

        # Adjust temperature, humidity, and air velocity to bring PMV closer to 0
        if current_pmv > target_pmv:
            room_temp -= 0.1  # Decrease temperature slightly
            air_velocity += 0.01  # Increase air velocity slightly
        else:
            room_temp += 0.1  # Increase temperature slightly
            air_velocity = max(0.1, air_velocity - 0.01)  # Decrease air velocity slightly, not below 0.1 m/s

        # Ensure conditions are within realistic limits
        room_temp = max(16, min(room_temp, 30))  # Temperature within 16°C to 30°C
        room_humidity = max(30, min(room_humidity, 70))  # Humidity within 30% to 70%
        air_velocity = max(0.1, min(air_velocity, 1.5))  # Air velocity within 0.1 m/s to 1.5 m/s

    return room_temp, room_humidity, air_velocity


def calculate_time_to_reach_temperature(current_temp, optimized_temp, room_area, room_volume, thermal_mass, heat_transfer_coefficient, ac_capacity):
    """
    Calculate the time required for the room to go from current temperature to optimized temperature.
    
    Parameters:
    - current_temp: Current room temperature (°C)
    - optimized_temp: Desired/optimized room temperature (°C)
    - room_area: Surface area of the room (m²)
    - room_volume: Volume of the room (m³)
    - thermal_mass: Thermal mass of the room (kJ/°C), typically calculated as room mass * specific heat capacity
    - heat_transfer_coefficient: Heat transfer coefficient (W/m²°C)
    - ac_capacity: Cooling capacity of the AC (kW)
    
    Returns:
    - time_to_reach_temp: Estimated time in hours to reach the optimized temperature
    """

    # Convert AC capacity from kW to kJ/h
    ac_capacity_kj_per_hour = ac_capacity * 3600  # 1 kW = 3600 kJ/h

    # Calculate the temperature difference
    delta_temp = abs(current_temp - optimized_temp)
    
    # Calculate the total heat that needs to be removed or added to reach the optimized temperature
    total_heat_change = thermal_mass * delta_temp  # in kJ

    # Consider the heat transfer rate due to the room's thermal mass and inertia
    heat_transfer_rate = heat_transfer_coefficient * room_area * delta_temp  # in W

    # Calculate the effective cooling rate
    effective_cooling_rate = ac_capacity_kj_per_hour - heat_transfer_rate  # in kJ/h

    # Estimate the time required to reach the optimized temperature
    if effective_cooling_rate <= 0:
        # If the cooling rate is not sufficient to cool down the room, return infinity
        return float('inf')
    
    time_to_reach_temp = total_heat_change / effective_cooling_rate  # in hours

    return time_to_reach_temp


def calculate_time_to_reach_temperature_with_air_velocity(current_temp, optimized_temp, room_area, room_volume, thermal_mass, heat_transfer_coefficient, ac_capacity, air_velocity):
    """
    Calculate the time required for the room to go from current temperature to optimized temperature, considering air velocity.
    
    Parameters:
    - current_temp: Current room temperature (°C)
    - optimized_temp: Desired/optimized room temperature (°C)
    - room_area: Surface area of the room (m²)
    - room_volume: Volume of the room (m³)
    - thermal_mass: Thermal mass of the room (kJ/°C)
    - heat_transfer_coefficient: Heat transfer coefficient (W/m²°C)
    - ac_capacity: Cooling capacity of the AC (kW)
    - air_velocity: Air velocity in the room (m/s)
    
    Returns:
    - time_to_reach_temp: Estimated time in hours to reach the optimized temperature
    """
    
    # Convert AC capacity from kW to kJ/h
    ac_capacity_kj_per_hour = ac_capacity * 3600  # 1 kW = 3600 kJ/h
    
    # Calculate the temperature difference
    delta_temp = abs(current_temp - optimized_temp)
    
    # Calculate the total heat that needs to be removed or added to reach the optimized temperature
    total_heat_change = thermal_mass * delta_temp  # in kJ
    
    # Modify the heat transfer coefficient based on air velocity
    adjusted_heat_transfer_coefficient = heat_transfer_coefficient * (1 + 0.5 * np.sqrt(air_velocity))
    
    # Consider the heat transfer rate due to the room's thermal mass and inertia
    heat_transfer_rate = adjusted_heat_transfer_coefficient * room_area * delta_temp  # in W
    
    # Calculate the effective cooling rate
    effective_cooling_rate = ac_capacity_kj_per_hour + heat_transfer_rate  # in kJ/h
    
    # Estimate the time required to reach the optimized temperature
    if effective_cooling_rate <= 0:
        # If the cooling rate is not sufficient to cool down the room, return infinity
        return float('inf')
    
    time_to_reach_temp = total_heat_change / effective_cooling_rate  # in hours

    return time_to_reach_temp


def calculate_ac_energy_consumption(current_temp, optimized_temp, current_humidity, optimized_humidity, room_volume, thermal_mass, ac_cop, ac_capacity):
    """
    Calculate the energy consumption of the AC to go from current temperature and humidity to an optimized setpoint.
    
    Parameters:
    - current_temp: Current room temperature (°C)
    - optimized_temp: Desired/optimized room temperature (°C)
    - current_humidity: Current room humidity (%)
    - optimized_humidity: Desired/optimized room humidity (%)
    - room_volume: Volume of the room (m³)
    - thermal_mass: Thermal mass of the room (kJ/°C)
    - ac_cop: Coefficient of performance of the AC (dimensionless)
    - ac_capacity: Cooling capacity of the AC (kW)
    
    Returns:
    - energy_consumption: Total energy consumption in kWh
    """

    # Calculate the temperature difference
    delta_temp = abs(current_temp - optimized_temp)
    
    # Calculate the heat energy required to change the temperature (sensible cooling)
    sensible_heat = thermal_mass * delta_temp  # in kJ

    # Calculate the latent heat required to remove humidity
    # Assumption: 1 m³ of air at 100% RH at 25°C contains 22.5 g of water vapor
    # Latent heat of vaporization of water is approximately 2,257 kJ/kg
    air_density = 1.2  # kg/m³, approximate air density at room temperature
    specific_humidity_current = 0.0225 * (current_humidity / 100)  # kg of water/kg of air
    specific_humidity_optimized = 0.0225 * (optimized_humidity / 100)  # kg of water/kg of air
    mass_of_air = room_volume * air_density  # Total mass of air in the room (kg)
    
    delta_humidity = mass_of_air * (specific_humidity_current - specific_humidity_optimized)  # kg of water
    latent_heat = delta_humidity * 2257  # Latent heat in kJ

    # Total cooling energy required (sensible + latent heat)
    total_cooling_energy = sensible_heat + latent_heat  # in kJ

    # Convert cooling energy from kJ to kWh
    total_cooling_energy_kwh = total_cooling_energy / 3600  # kWh

    # Calculate energy consumption based on AC COP
    energy_consumption = total_cooling_energy_kwh / ac_cop  # in kWh

    return energy_consumption


def calculate_feels_like_temperature(actual_temp, humidity, air_velocity, metabolic_rate=1.0, skin_temp=33.0):
    """
    Calculate the feels-like temperature by incorporating air velocity, humidity, actual temperature, metabolic rate, and skin temperature.
    
    Parameters:
    - actual_temp: Actual air temperature in °C
    - humidity: Relative humidity in %
    - air_velocity: Air velocity in m/s
    - metabolic_rate: Metabolic rate in met (1 met = 58.2 W/m², default is 1.0 for resting)
    - skin_temp: Skin temperature in °C (default is 33°C)
    
    Returns:
    - feels_like_temp: Feels-like temperature in °C
    """

    # Adjust for humidity effect (simplified heat index formula)
    if actual_temp >= 27:
        heat_index = actual_temp + 0.33 * humidity - 0.7 * air_velocity - 4.0
    else:
        heat_index = actual_temp

    # Adjust for wind chill effect
    if actual_temp <= 10 and air_velocity >= 1.3:
        wind_chill = 13.12 + 0.6215 * actual_temp - 11.37 * (air_velocity ** 0.16) + 0.3965 * actual_temp * (air_velocity ** 0.16)
        feels_like_temp = wind_chill
    else:
        feels_like_temp = heat_index

    # Adjust for metabolic rate (higher metabolic rate increases the feels-like temperature)
    # Assume a linear relationship: higher metabolic rate increases heat production, hence higher feels-like temperature.
    feels_like_temp += (metabolic_rate - 1.0) * 2  # Adjust based on deviation from 1.0 met

    # Adjust for skin temperature (lower skin temperature decreases the feels-like temperature)
    # Assume a linear relationship: lower skin temperature indicates greater heat loss, reducing feels-like temperature.
    feels_like_temp += (skin_temp - 33.0) * 0.5  # Adjust based on deviation from normal skin temperature (33°C)

    return feels_like_temp


def optimize_feels_like_temperature(actual_temp, humidity, metabolic_rate, skin_temp, optimized_temp, max_air_velocity=3.0):
    """
    Adjust the air velocity to make the feels-like temperature as close as possible to the optimized temperature.
    
    Parameters:
    - actual_temp: Actual air temperature in °C
    - humidity: Relative humidity in %
    - metabolic_rate: Metabolic rate in met (1 met = 58.2 W/m²)
    - skin_temp: Skin temperature in °C
    - optimized_temp: Desired/optimized feels-like temperature in °C
    - max_air_velocity: Maximum air velocity in m/s (default is 3.0 m/s)
    
    Returns:
    - optimal_air_velocity: The air velocity (m/s) needed to achieve the desired feels-like temperature
    - final_feels_like_temp: The resulting feels-like temperature
    """
    current_air_velocity = 0.1  # Start with a minimum air velocity
    tolerance = 0.1  # Tolerance level for how close the feels-like temperature should be to the optimized temperature
    max_iterations = 100  # Limit iterations to avoid infinite loops

    for _ in range(max_iterations):
        feels_like_temp = calculate_feels_like_temperature(actual_temp, humidity, current_air_velocity, metabolic_rate, skin_temp)
        
        # Check if the feels-like temperature is within the desired range
        if abs(feels_like_temp - optimized_temp) <= tolerance:
            break
        
        # Increase air velocity to try to match the optimized temperature
        if feels_like_temp > optimized_temp:
            current_air_velocity = min(current_air_velocity + 0.1, max_air_velocity)
        else:
            break  # If increasing air velocity doesn't help, stop the loop
    
    final_feels_like_temp = calculate_feels_like_temperature(actual_temp, humidity, current_air_velocity, metabolic_rate, skin_temp)
    
    return current_air_velocity, final_feels_like_temp

# Example usage:
# actual_temp = 30  # Actual temperature in °C
# humidity = 60  # Relative humidity in %
metabolic_rate = 1.2  # Metabolic rate in met (e.g., light work)
skin_temp = 32.5  # Skin temperature in °C
# optimized_temp = 27  # Desired/optimized feels-like temperature in °C







# Example usage:
current_temp = 28  # Current room temperature in °C
optimized_temp = 24  # Desired/optimized room temperature in °C
current_humidity = 60  # Current room humidity in %
optimized_humidity = 50  # Desired/optimized room humidity in %
room_volume = 150  # Volume of the room in m³
thermal_mass = 1000  # Thermal mass of the room in kJ/°C
ac_cop = 3.5  # Coefficient of performance of the AC (dimensionless)
ac_capacity = 3.5  # AC cooling capacity in kW

energy_consumption = calculate_ac_energy_consumption(current_temp, optimized_temp, current_humidity, optimized_humidity, room_volume, thermal_mass, ac_cop, ac_capacity)

print(f"AC Energy Consumption: {energy_consumption:.2f} kWh")

# Example usage:
forecast_temp = 35  # Example forecasted temperature
forecast_humidity = 60  # Example forecasted humidity
current_temp = 28  # Example current room temperature
current_humidity = 50  # Example current room humidity
current_velocity = 0.1  # Example current air velocity

optimized_temp, optimized_humidity, optimized_velocity = optimize_conditions_to_pmv_zero(
    forecast_temp, forecast_humidity, current_temp, current_humidity, current_velocity)

# In a table format, show input and output values
print("Input Conditions:")
print(f"Forecasted Temperature: {forecast_temp} °C")
print(f"Forecasted Humidity: {forecast_humidity}%")
print(f"Current Room Temperature: {current_temp} °C")
print(f"Current Room Humidity: {current_humidity}%")
print(f"Current Air Velocity: {current_velocity} m/s")
print("\nOptimized Conditions:")
print(f"Optimized Room Temperature: {optimized_temp:.1f} °C")
print(f"Optimized Room Humidity: {optimized_humidity}%")
print(f"Optimized Air Velocity: {optimized_velocity} m/s")

# Example usage:
# current_temp = 28  # Current room temperature in °C
# optimized_temp = 24  # Desired/optimized room temperature in °C
room_area = 50  # Surface area of the room in m²
room_volume = 150  # Volume of the room in m³
thermal_mass = 1000  # Thermal mass of the room in kJ/°C
heat_transfer_coefficient = 5  # Heat transfer coefficient in W/m²°C
ac_capacity = 3.5  # AC cooling capacity in kW
air_velocity = 1

time_to_temp = calculate_time_to_reach_temperature(current_temp, optimized_temp, room_area, room_volume, thermal_mass, heat_transfer_coefficient, ac_capacity)
time_to_temp_air_velocity = calculate_time_to_reach_temperature_with_air_velocity(current_temp, optimized_temp, room_area, room_volume, thermal_mass, heat_transfer_coefficient, ac_capacity, air_velocity)
energy_consumption = calculate_ac_energy_consumption(current_temp, optimized_temp, current_humidity, optimized_humidity, room_volume, thermal_mass, ac_cop, ac_capacity)
optimal_air_velocity, final_feels_like_temp = optimize_feels_like_temperature(current_temp, current_humidity, metabolic_rate, skin_temp, optimized_temp)
# Initial feels-like temperature
initial_feels_like_temp = calculate_feels_like_temperature(current_temp, current_humidity, air_velocity, metabolic_rate, skin_temp)
# Initial PMV
initial_pmv = calculate_pmv(current_temp, current_temp, current_humidity, met=metabolic_rate, air_velocity=air_velocity)
print(f"Time to reach optimized temperature: {time_to_temp:.2f} hours")
print(f"Time to reach optimized temperature with air velocity: {time_to_temp_air_velocity:.2f} hours")
print(f"AC Energy Consumption: {energy_consumption:.2f} kWh")
print(f"Optimal Air Velocity: {optimal_air_velocity:.2f} m/s")
print(f"Initial Feels Like Temperature: {initial_feels_like_temp:.2f}°C")
print(f"Initial PMV Value: {initial_pmv:.2f}")
print(f"Final Feels Like Temperature: {final_feels_like_temp:.2f}°C")
pmv_value = calculate_pmv(current_temp, current_temp, current_humidity, met=metabolic_rate, air_velocity=optimal_air_velocity)
print(f"Final PMV Value: {pmv_value:.2f}")