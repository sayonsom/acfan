import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the generated data
input_file = 'synthetic_pmv_data_500000_1725514933.csv'
data_size = input_file.split("_")[2]
synthetic_data_version = input_file.split("_")[3].replace(".csv", "")
data = pd.read_csv(input_file)

# Separate features and target
X = data[['PMV', 'HeartRate', 'Clo', 'Humidity', 'SkinTemperature']]  # Added SkinTemperature
y = data['AirVelocity_AirTemperature'].apply(lambda x: eval(x))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target tuples to numpy arrays
y_train_np = np.array(y_train.tolist())
y_test_np = np.array(y_test.tolist())

# Define model building function
def build_model(units_1=64, units_2=32, dropout_1=0.2, dropout_2=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(units_1, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(dropout_1),
        Dense(units_2, activation='relu'),
        Dropout(dropout_2),
        Dense(2)  # Output layer with 2 units for Air Velocity and Air Temperature
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
    return model

# Define hyperparameter search space
param_dist = {
    'model__units_1': [64, 128, 256],
    'model__units_2': [32, 64, 128],
    'model__dropout_1': [0.1, 0.2, 0.3],
    'model__dropout_2': [0.1, 0.2, 0.3],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 150]
}

# Create KerasRegressor model
model = KerasRegressor(model=build_model, verbose=0)

# Perform RandomizedSearchCV
n_iter_search = 20
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter_search, cv=3, verbose=2, n_jobs=-1)

# Fit RandomizedSearchCV
random_search.fit(X_train_scaled, y_train_np)

# Get best model
best_model = random_search.best_estimator_.model_

# Evaluate model
mae = best_model.evaluate(X_test_scaled, y_test_np)[1]
print(f'Best MAE: {mae}')

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Compare predictions with actual values
comparison = pd.DataFrame({
    'Actual_AirVelocity': y_test_np[:, 0],
    'Predicted_AirVelocity': y_pred[:, 0],
    'Actual_AirTemperature': y_test_np[:, 1],
    'Predicted_AirTemperature': y_pred[:, 1]
})
print(comparison.head())

# Save the model
best_model.save(f'models/combined_model_{data_size}_{synthetic_data_version}.keras')

# Save the scaler
joblib.dump(scaler, f'scalers/scaler_{data_size}_{synthetic_data_version}.save')

# Retrain the best model to get the history
best_params = random_search.best_params_

# Function to get model parameters
def get_model_params(params):
    return {k.split('__')[1]: v for k, v in params.items() if k.startswith('model__')}

# Retrain model
model = build_model(**get_model_params(best_params))
history = model.fit(
    X_train_scaled, y_train_np,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_split=0.2,
    verbose=0
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Combined Model (Air Velocity and Air Temperature)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'models/plots/training_history_{data_size}_{synthetic_data_version}.png')
plt.show()

print("Training completed. Model and plot saved.")