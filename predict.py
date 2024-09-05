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
    min_temp, max_temp = predicted_temp - 10, predicted_temp + 10  # Wider search range
    
    for _ in range(max_iterations):
        mid_temp = (min_temp + max_temp) / 2
        
        # Calculate PMV using the current temperature guess and target velocity
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

# Example usage
if __name__ == "__main__":
    # Load the combined model and scaler
    data_size = "500000"  
    synthetic_data_version = "1725514933"  
    combined_model, scaler = load_model_and_scaler(data_size, synthetic_data_version)
    
    # Just some trial data to start with
    example_data = np.array([
        [0.50, 75, 33.0, 1.0, 50],  # Example 1
        [0.61, 80, 32.5, 0.8, 60],  # Example 2
    ])
    
    target_velocity = 0.1  # Define target velocity
    
    results = []
    for data in example_data:
        pmv, heart_rate, skin_temp, clo, humidity = data
        
        # Prediction without constraint
        predicted_temp_unconstrained, predicted_velocity = predict_temp_and_velocity(
            combined_model, scaler, pmv, heart_rate, skin_temp, clo, humidity
        )
        
        # Prediction with target air velocity constraint
        predicted_temp_constrained = find_air_temperature_with_constraint(
            pmv, clo, humidity, 
            predicted_temp_unconstrained, predicted_velocity, target_velocity=target_velocity
        )
        
        # Determine final setpoints as a tuple
        if predicted_temp_unconstrained > predicted_temp_constrained:
            final_setpoints = (predicted_temp_unconstrained, predicted_velocity)
        else:
            final_setpoints = (predicted_temp_constrained, target_velocity)
        
        results.append({
            'PMV': pmv,
            'HeartRate': heart_rate,
            'SkinTemp': skin_temp,
            'Clo': clo,
            'Humidity': humidity,
            'PredictedTempWithTargetAirVelocityConstraint': predicted_temp_constrained,
            'PredictedTempWithoutConstraint': predicted_temp_unconstrained,
            'PredictedAirVelocity': predicted_velocity,
            'FinalSetpoints': final_setpoints
        })
    
    # Create a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    
    print("Comparison of Predictions and Final Setpoints:")
    print(results_df)
    
    # Display the final setpoints more clearly
    print("\nFinal Setpoints (Temperature, Air Velocity):")
    for index, row in results_df.iterrows():
        print(f"Example {index + 1}: {row['FinalSetpoints']}")