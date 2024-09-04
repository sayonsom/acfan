import math
from pythermalcomfort.models import pmv_ppd

# Constants
surface_area_walls = 50  # Surface area of walls/windows in m² (estimated for a room)
specific_heat_air = 1.005  # kJ/kg°C for dry air
air_density = 1.225  # kg/m³ at 15°C
max_ac_power = 3500  # Maximum power of the air conditioner in Watts
time_step = 60  # Simulation time step in seconds

# Function to estimate indoor temperature change due to outdoor temperature
def estimate_indoor_temp(T_in, T_out, room_volume, insulation_R, time_hours):
    # Calculate heat transfer coefficient
    U = 1 / insulation_R
    
    # Calculate room's thermal mass
    room_mass_air = room_volume * air_density
    thermal_mass = room_mass_air * specific_heat_air
    
    # Calculate time constant (how quickly the room responds to temperature changes)
    time_constant = thermal_mass / (U * surface_area_walls)
    
    # Calculate the new indoor temperature using an exponential approach
    temperature_difference = T_out - T_in
    new_T_in = T_out - temperature_difference * math.exp(-time_hours / time_constant)
    
    return new_T_in

# Function to calculate PMV using pythermalcomfort
def calculate_pmv(T_in, humidity, air_velocity=0.1, met=1.0, clo=0.5):
    # PMV calculation based on indoor conditions
    result = pmv_ppd(tdb=T_in, tr=T_in, rh=humidity, vr=air_velocity, met=met, clo=clo)
    return result['pmv'], result['ppd']


def estimate_energy_consumption(T_in, T_setpoint, T_out, room_volume, insulation_R, hours):
    U = 1 / insulation_R
    room_mass_air = room_volume * air_density
    thermal_mass = room_mass_air * specific_heat_air
    
    results = []
    T_in_no_ac = T_in
    
    for hour in range(hours):
        # Calculate heat gain from the environment
        heat_gain = U * surface_area_walls * (T_out - T_in) * 3600  # 3600 seconds in an hour
        
        # Calculate energy needed to cool the room
        energy_needed = thermal_mass * (T_in - T_setpoint) * 1000  # Convert to Joules
        
        # Total energy required (cooling + offsetting heat gain)
        total_energy = max(0, energy_needed + heat_gain)  # Ensure non-negative
        
        # Convert Joules to kWh
        energy_kwh = total_energy / 3600000
        
        # Calculate indoor temperature without AC
        T_in_no_ac = estimate_indoor_temp(T_in_no_ac, T_out, room_volume, insulation_R, 1)
        
        # Indoor temperature with AC (equal to setpoint)
        T_in_with_ac = T_setpoint
        
        # Calculate PMV for the room with AC
        pmv, _ = calculate_pmv(T_in_with_ac+4, humidity=50, air_velocity=0.58, met=1.0, clo=0.5)
        
        results.append({
            'hour': hour + 1,
            'energy_kwh': energy_kwh,
            'T_out': T_out,
            'T_in_no_ac': T_in_no_ac,
            'T_in_with_ac': T_in_with_ac,
            'pmv': pmv
        })
        
        # Update indoor temperature for next hour (assuming AC maintains setpoint)
        T_in = T_setpoint
    
    return results


# Input values
T_in = 25.0  # Initial indoor temperature in °C
humidity_in = 50  # Indoor relative humidity in %
T_out = 35.0  # Outdoor temperature in °C
humidity_out = 40  # Outdoor relative humidity in %
room_volume = 60  # Room volume in m³ (example for a medium-sized room)
insulation_R = 2.0  # Insulation R-value (m²·K/W)
time_hours = 1.0  # Time duration in hours for temperature estimation

# Estimate indoor temperature after the given time
new_T_in = estimate_indoor_temp(T_in, T_out, room_volume, insulation_R, time_hours)
print(f"Estimated Indoor Temperature after {time_hours} hours: {new_T_in:.2f}°C")

# Calculate PMV for the estimated indoor temperature
pmv, ppd = calculate_pmv(new_T_in, humidity_in)
print(f"PMV: {pmv:.2f}, PPD: {ppd:.2f}%")

# Demonstrate temperature change over time
print("\nTemperature change over time:")
for hour in range(1, 13):
    temp = estimate_indoor_temp(T_in, T_out, room_volume, insulation_R, hour)
    print(f"After {hour} hours: {temp:.2f}°C")


# Input values
T_in = 28.0  # Initial indoor temperature in °C
humidity_in = 50  # Indoor relative humidity in %
T_out = 35.0  # Outdoor temperature in °C
room_volume = 60  # Room volume in m³
insulation_R = 2.0  # Insulation R-value (m²·K/W)
AC_setpoint = 24.0  # AC setpoint temperature in °C

# Estimate energy consumption and other parameters for 24 hours
results = estimate_energy_consumption(T_in, AC_setpoint, T_out, room_volume, insulation_R, 24)

print("\nEstimated Energy Consumption and Thermal Comfort Metrics:")
print("Hour | Energy (kWh) | Outdoor Temp (°C) | Indoor Temp w/o AC (°C) | Indoor Temp w/ AC (°C) | PMV")
print("-" * 95)
for result in results:
    print(f"{result['hour']:2d}   | {result['energy_kwh']:6.2f}      | {result['T_out']:8.2f}         | {result['T_in_no_ac']:11.2f}            | {result['T_in_with_ac']:11.2f}           | {result['pmv']:5.2f}")

total_energy = sum(result['energy_kwh'] for result in results)
print(f"\nTotal energy consumption over 24 hours: {total_energy:.2f} kWh")