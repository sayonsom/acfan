import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import json
from pythermalcomfort.models import pmv

def predict_ml(total_occupants: int, indoor_temp: float, outdoor_temp: float,
               humidity: float, desired_comfort_temp: float, ac_type: str = "windfree",
               met: float = 1.2, clo: float = 0.5) -> dict:
    """
    Predict optimal AC temperature and fan settings using ML model.
    
    Args:
        total_occupants: Number of occupants in the room
        indoor_temp: Current indoor temperature in Celsius
        outdoor_temp: Outdoor temperature in Celsius
        humidity: Current humidity percentage (0-100)
        desired_comfort_temp: Target feel-like temperature in Celsius
        ac_type: Type of AC ("windfree" or other)
        met: Metabolic rate (default 1.2)
        clo: Clothing insulation (default 0.5)
    
    Returns:
        dict: JSON containing input parameters, predictions, and status
    """
    try:
        # Input validation
        if desired_comfort_temp < 16 or desired_comfort_temp > 32:
            raise ValueError("Desired temperature must be between 16 and 32°C")
        if outdoor_temp < desired_comfort_temp:
            raise ValueError("Heating feature is not supported")

        # Adjust humidity based on AC type
        effective_humidity = humidity if ac_type.lower() == "windfree" else min(humidity + 5, 100)
        
        # Current timestamp
        current_time = datetime.now()

        # Calculate target PMV
        pmv_baseline_temp = 24.0
        pmv_temp_sensitivity = 0.3
        target_pmv = (desired_comfort_temp - pmv_baseline_temp) * pmv_temp_sensitivity

        # Initialize response dictionary
        response = {
            "timestamp": current_time.isoformat(),
            "input_parameters": {
                "total_occupants": total_occupants,
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
            "recheck_needed": False,
            "ml_model_info": {
                "prediction_confidence": None,
                "model_version": "best_model.keras"
            }
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

        # Prepare input data for ML model
        input_data = pd.DataFrame([[
            total_occupants, indoor_temp, outdoor_temp, 
            effective_humidity, target_pmv, met, clo
        ]], columns=[
            'Total_Occupants', 'Actual_Indoor_Temp', 'Outdoor_Temperature',
            'Humidity', 'PMV_Target', 'Metabolic_Rate', 'Clothing_Insulation'
        ])

        # Load scaler and model
        scaler = joblib.load('scalers/scaler_X.joblib')
        model = load_model('models/best_model.keras')

        # Make predictions
        scaled_input = scaler.transform(input_data)
        setpoint_pred, velocity_pred = model.predict(scaled_input, verbose=0)
        
        # Extract predictions
        ac_setpoint = float(setpoint_pred[0])
        air_velocity = float(velocity_pred[0])

        # Map air velocity to fan speed
        fan_speeds = {
            120: 1, 165: 2, 210: 3, 250: 4, 295: 5, 335: 6
        }
        
        fan_rpm = None
        fan_speed = None
        for rpm, speed in fan_speeds.items():
            calculated_velocity = (rpm / 335) * 3.2
            if calculated_velocity >= air_velocity:
                fan_rpm = rpm
                fan_speed = speed
                break

        # Calculate delivered PMV with predicted settings
        delivered_pmv = pmv(tdb=ac_setpoint, tr=ac_setpoint, 
                           vr=air_velocity, rh=effective_humidity, 
                           met=met, clo=clo)

        # Update response with predictions
        response["ac_setpoint"] = round(ac_setpoint, 1)
        response["fan_speed"] = fan_speed
        response["fan_rpm"] = fan_rpm
        response["pmv_calculations"]["delivered_pmv"] = round(delivered_pmv, 3)
        response["ml_model_info"]["prediction_confidence"] = round(float(np.mean([
            model.predict(scaled_input, verbose=0)[0][0],
            model.predict(scaled_input, verbose=0)[1][0]
        ])), 3)

        return response

    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Example with high humidity
        result = predict_ml(
            total_occupants=2,
            indoor_temp=28,
            outdoor_temp=32,
            humidity=80,
            desired_comfort_temp=24,
            ac_type="windfree"
        )
        print("High Humidity Response:")
        print(json.dumps(result, indent=2))
        
        print("\n" + "-"*50 + "\n")
        
        # # Example with normal humidity
        # result = predict_ml(
        #     total_occupants=2,
        #     indoor_temp=28,
        #     outdoor_temp=32,
        #     humidity=60,
        #     desired_comfort_temp=24,
        #     ac_type="windfree"
        # )
        # print("Normal Humidity Response:")
        # print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")