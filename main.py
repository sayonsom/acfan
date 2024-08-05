from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from utilities.conversions import fan_speed_to_air_velocity

app = FastAPI()
templates = Jinja2Templates(directory="templates")

data_size = 500000
synthetic_data_version = 1722757024
model_name = f'models/synthetic_pmv_data_{data_size}_{synthetic_data_version}.h5'
model = tf.keras.models.load_model(model_name)
scaler = joblib.load(f'scalers/scaler_{data_size}_{synthetic_data_version}.save')

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

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  outdoor_temp: float = Form(...),
                  humidity: float = Form(...),
                  clo_level: int = Form(...),
                  ac_tonnage: float = Form(...),
                  fan_speed: int = Form(...),
                  floor_length_ft: float = Form(...),
                  floor_width_ft: float = Form(...),
                  ceiling_height_ft: float = Form(...),
                  num_windows: int = Form(...),
                  window_height_ft: float = Form(...),
                  window_width_ft: float = Form(...),
                  window_facing: str = Form(...),
                  curtains: str = Form(...)):
    clo_values = [0.5, 1.0, 0.5, 0.7, 0.2, 0.5, 1.0, 1.5, 1.2, 1.5, 2.0]
    clo = clo_values[clo_level - 1]
    curtains_on = curtains.lower() == "yes"
    
    ac_cop = 3  # coefficient of performance of the AC
    fan_power = 0.075  # power of the ceiling fan in kW (75 watts)
    indoor_temp = outdoor_temp - 4.5
    initial_temp = indoor_temp  # initial room temperature in °C
    pmv = 0
    ac_power = ac_tonnage * 3.5
    air_velocity = fan_speed_to_air_velocity(fan_speed, ceiling_height=ceiling_height_ft, room_area=floor_length_ft * floor_width_ft)
    
    floor_length = convert_ft_to_m(floor_length_ft)
    floor_width = convert_ft_to_m(floor_width_ft)
    ceiling_height = convert_ft_to_m(ceiling_height_ft)
    window_height = convert_ft_to_m(window_height_ft)
    window_width = convert_ft_to_m(window_width_ft)
    
    room_area = floor_length * floor_width
    window_area = estimate_window_area(num_windows, window_height, window_width)
    
    wall_thermal_resistance = 0.5
    window_thermal_resistance = 0.8 if not curtains_on else 1.5
    
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

    mrt = calculate_MRT(surface_temps, surface_areas, window_facing, curtains_on)
    
    data = {
        'RadiantTemperature': [mrt],
        'AirVelocity': [air_velocity],
        'Humidity': [humidity],
        'Clo': [clo],
        'PMV': [pmv]
    }
    df = pd.DataFrame(data)
    input_features = df[['RadiantTemperature', 'AirVelocity', 'Humidity', 'Clo', 'PMV']]
    input_scaled = scaler.transform(input_features)
    predicted_air_temperature = model.predict(input_scaled)

    target_temp = predicted_air_temperature[0][0]

    ac_energy, ac_time, ac_fan_energy, ac_fan_time = predict_cooling_time_and_energy(ac_power, ac_cop, fan_power, initial_temp, target_temp, room_volume)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": f"{target_temp:.2f}°C",
        "ac_energy": f"{ac_energy:.2f} kWh",
        "ac_time": f"{ac_time:.2f} hours",
        "ac_fan_energy": f"{ac_fan_energy:.2f} kWh",
        "ac_fan_time": f"{ac_fan_time:.2f} hours"
    })

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
