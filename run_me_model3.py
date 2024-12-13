from pythermalcomfort.models import pmv
import numpy as np
from datetime import datetime, timedelta
import json

def predict(humidity: float, outdoor_temp: float, indoor_temp: float, 
            desired_comfort_temp: float, ac_type: str = "windfree") -> dict:
    """
    Predict optimal AC temperature and fan settings for thermal comfort.
    
    Args:
        humidity: Current humidity percentage (0-100)
        outdoor_temp: Outdoor temperature in Celsius
        indoor_temp: Current indoor temperature in Celsius
        desired_comfort_temp: Target feel-like temperature in Celsius
        ac_type: Type of AC ("windfree" or other)
    
    Returns:
        dict: JSON containing input parameters, status message, settings, and PMV values
    """
    # Input validation
    if desired_comfort_temp < 16 or desired_comfort_temp > 32:
        raise ValueError("Desired temperature must be between 16 and 32°C")
    if outdoor_temp < desired_comfort_temp:
        raise ValueError("Heating feature is not supported")

    # Adjust humidity based on AC type
    effective_humidity = humidity if ac_type.lower() == "windfree" else min(humidity + 5, 100)
    
    # Current timestamp
    current_time = datetime.now()
    
    # Constants for calculation
    met = 1.2  # Assuming light activity
    clo = 0.5  # Assuming summer clothing
    pmv_baseline_temp = 24.0
    pmv_temp_sensitivity = 0.3

    # Calculate target PMV
    target_pmv = (desired_comfort_temp - pmv_baseline_temp) * pmv_temp_sensitivity
    
    # Initialize response dictionary with input parameters
    response = {
        "timestamp": current_time.isoformat(),
        "input_parameters": {
            "humidity": humidity,
            "effective_humidity": effective_humidity,
            "outdoor_temp": outdoor_temp,
            "indoor_temp": indoor_temp,
            "desired_comfort_temp": desired_comfort_temp,
            "ac_type": ac_type,
            "metabolic_rate": met,
            "clothing_insulation": clo
        },
        "pmv_calculations": {
            "target_pmv": round(target_pmv, 3),
            "delivered_pmv": None
        },
        "status_message": "",
        "next_check": None,
        "ac_setpoint": None,
        "fan_speed": None,
        "fan_rpm": None,
        "recheck_needed": False
    }

    # High humidity handling
    if effective_humidity > 75:
        response["status_message"] = ("To make sure you are comfortable will first reduce the humidity "
                                    "a little bit, and then automatically turn on the Efficient AI cooling mode. "
                                    "Sit back and relax")
        response["recheck_needed"] = True
        response["next_check"] = (current_time + timedelta(minutes=15)).isoformat()
        response["ac_setpoint"] = desired_comfort_temp
        
        # Calculate initial PMV with just AC (no fan)
        initial_pmv = pmv(tdb=desired_comfort_temp, tr=desired_comfort_temp, 
                         vr=0.1, rh=effective_humidity, met=met, clo=clo)
        response["pmv_calculations"]["delivered_pmv"] = round(initial_pmv, 3)
        return response

    response["status_message"] = "Starting Efficient AI Cooling Now"
    
    # Define search ranges
    ac_temp_range = np.arange(28.0, desired_comfort_temp - 0.5, -0.5)
    air_velocity_range = np.arange(0.1, 3.3, 0.1)

    # Fan speed mapping -- #TODO: NEEDS TO update based on what we hear back from polycab
    fan_speeds = {
        120: 1,  # RPM: Speed
        165: 2,
        210: 3,
        250: 4,
        295: 5,
        335: 6
    }

    # Find optimal combination
    best_ac_temp = None
    best_air_velocity = None
    closest_pmv = float('inf')

    for air_velocity in reversed(air_velocity_range):
        for ac_temp in ac_temp_range:
            pmv_value = pmv(tdb=ac_temp, tr=ac_temp, vr=air_velocity, 
                           rh=effective_humidity, met=met, clo=clo)
            
            if abs(pmv_value - target_pmv) < abs(closest_pmv - target_pmv):
                best_ac_temp = ac_temp
                best_air_velocity = air_velocity
                closest_pmv = pmv_value

    if best_air_velocity is None or best_ac_temp is None:
        raise ValueError("Could not determine optimal settings")

    # Map air velocity to fan speed
    fan_rpm = None
    fan_speed = None
    for rpm, speed in fan_speeds.items():
        calculated_velocity = (rpm / 335) * 3.2  # Max RPM (335) produces max velocity (3.2 m/s)
        if calculated_velocity >= best_air_velocity:
            fan_rpm = rpm
            fan_speed = speed
            break

    # Update response with calculated values
    response["ac_setpoint"] = round(best_ac_temp, 1)
    response["fan_speed"] = fan_speed
    response["fan_rpm"] = fan_rpm
    response["pmv_calculations"]["delivered_pmv"] = round(closest_pmv, 3)
    
    return response

# Example usage
if __name__ == "__main__":
    try:
        # Example with high humidity
        result = predict(
            humidity=80,
            outdoor_temp=32,
            indoor_temp=28,
            desired_comfort_temp=24,
            ac_type="otherwindfree"
        )

        # Normal humidity -- uncomment for testing
        # result = predict(
        #     humidity=60,
        #     outdoor_temp=32,
        #     indoor_temp=28,
        #     desired_comfort_temp=24,
        #     ac_type="windfree"
        # )

        print("High Humidity Response:")
        print(json.dumps(result, indent=2))
        
        print("\n" + "-"*50 + "\n")
        
        

            
    except ValueError as e:
        print(f"Error: {str(e)}")