from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

data_size = 500000
synthetic_data_version = 1722757024
model_name = f'models/synthetic_pmv_data_{data_size}_{synthetic_data_version}.h5'
model = tf.keras.models.load_model(model_name)
scaler = joblib.load(f'scalers/scaler_{data_size}_{synthetic_data_version}.save')

def calculate_surface_temperature(T_i, T_o, R_i, R_o):
    return (T_i * R_i + T_o * R_o) / (R_i + R_o)

def calculate_MRT(surface_temps, surface_areas):
    total_area = sum(surface_areas)
    weighted_temp = sum(T * A for T, A in zip(surface_temps, surface_areas))
    return weighted_temp / total_area

def estimate_window_area(num_windows, window_height, window_width):
    return num_windows * window_height * window_width

def convert_ft_to_m(feet):
    return feet * 0.3048

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, pmv: float = Form(...), humidity: float = Form(...), clo: float = Form(...), air_velocity: float = Form(...), mrt: float = Form(...)):
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

    return templates.TemplateResponse("index.html", {"request": request, "prediction": f"{predicted_air_temperature[0][0]:.2f}"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
