import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import joblib

# Load the generated data
input_file = 'synthetic_pmv_data_500000_1722757024.csv'
data_size = input_file.split("_")[3]
synthetic_data_version = input_file.split("_")[4].replace(".csv", "")
data = pd.read_csv(input_file)

# Separate features and target
X = data.drop('AirTemperature', axis=1)
y = data['AirTemperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

# Create a 1x2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training history (Loss)
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot MAE history
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean Absolute Error')
ax2.legend()

# Adjust layout and save the combined plot
plt.suptitle(f'Training History: Version {synthetic_data_version}, LOSS: {loss}, MAE: {mae}')
plt.tight_layout()
plt.savefig(f'models/plots/synthetic_pmv_data_{data_size}_{synthetic_data_version}.png')
plt.show()


print(f'Test Loss: {loss}')
print(f'Test MAE: {mae}')

# Make predictions
y_pred = model.predict(X_test)

# Save the model as h5 file based on the data file name, after the third underscore
model_name = f'models/synthetic_pmv_data_{data_size}_{synthetic_data_version}.h5'
model.save(model_name)

# Save the scaler
joblib.dump(scaler, f'scalers/scaler_{data_size}_{synthetic_data_version}.save')

# Compare predictions with actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(comparison.head())
