import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# Read the CSV file
df = pd.read_csv('thermal_comfort_data.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# Create a custom colormap for thermal comfort
colors = ['blue', 'green', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('thermal_comfort', colors, N=n_bins)

# Create the plot
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig.suptitle('24-Hour Thermal Comfort and Physiological Data', fontsize=16, y=0.95)

# Plot 1: Temperature and AC Setpoint
ax1 = axs[0]
ax1.plot(df['Time'], df['Outdoor Temp'], label='Outdoor Temp', color='red')
ax1.plot(df['Time'], df['AC_Setpoint'], label='AC Setpoint', color='blue')
ax1.set_ylabel('Temperature (°C)')
ax1.legend(loc='upper left')
ax1.set_title('Temperature and AC Setpoint', pad=10)

# Add Air Velocity to the same plot with a secondary y-axis
ax1_twin = ax1.twinx()
ax1_twin.plot(df['Time'], df['Air Velocity'], label='Air Velocity', color='green', linestyle='--')
ax1_twin.set_ylabel('Air Velocity (m/s)')
ax1_twin.legend(loc='upper right')

# Plot 2: Energy Consumption
ax2 = axs[1]
ax2.plot(df['Time'], df['Energy Consumed_WithAirVelocityAdjustment'], 
         label='With Air Velocity Adjustment', color='green')
ax2.plot(df['Time'], df['EnergyConsumed_WithoutAirVelocity'], 
         label='Without Air Velocity', color='orange')
ax2.set_ylabel('Energy Consumed (units)')
ax2.legend(loc='upper left')
ax2.set_title('Energy Consumption Comparison', pad=10)

# Plot 3: Physiological Data and Thermal Comfort
ax3 = axs[2]
sc = ax3.scatter(df['Time'], df['Heart Rate'], c=df['PMV_Thermal Comfort'], 
                 cmap=cmap, s=30, label='Heart Rate')
ax3.set_ylabel('Heart Rate (bpm)')
ax3.set_title('Physiological Data and Thermal Comfort', pad=10)

# Add Skin Temperature to the same plot with a secondary y-axis
ax3_twin = ax3.twinx()
ax3_twin.plot(df['Time'], df['Skin Temperature'], color='red', label='Skin Temperature')
ax3_twin.set_ylabel('Skin Temperature (°C)')

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Add colorbar for thermal comfort
cbar = fig.colorbar(sc, ax=ax3, label='PMV Thermal Comfort')
cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cbar.set_ticklabels(['Cold', 'Cool', 'Slightly Cool', 'Neutral', 'Slightly Warm', 'Warm', 'Hot'])

# Format x-axis to show hours
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.grid(True, linestyle='--', alpha=0.7)

plt.xlabel('Time of Day')
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Adjust the top margin
plt.savefig('thermal_comfort_plot.png')